"""Transcript translator for Harry Potter audiobook materials."""

import hashlib
import time
from typing import TYPE_CHECKING, List

from loguru import logger

from video_gen.common.tools import llm_disk_cache_load, llm_disk_cache_save
from video_gen.video_material import TranscriptSegment

# Bump when TRANSLATION_PROMPT text changes to invalidate on-disk caches.
TRANSLATION_PROMPT_VERSION = 1

if TYPE_CHECKING:
    from video_gen.core.tools.openai_client import OpenAIClient


# Translation prompt template
TRANSLATION_PROMPT = """You are an expert English-to-Chinese translator specializing in Harry Potter books.

You will receive:
1. English Transcript Segments: A numbered list of English text segments to translate
2. Chinese Reference Text: The official Chinese translation of the Harry Potter chapter for reference

Your task: Translate EACH English segment to Chinese while maintaining consistency with the official translation.

🚨 CRITICAL LINE COUNT REQUIREMENT 🚨
- Your output MUST have EXACTLY the same number of lines as the input segments
- If input has 288 segments, output MUST have EXACTLY 288 lines (not 287, not 289, EXACTLY 288)
- Each line in your output corresponds to ONE segment from the input
- Count your output lines carefully before submitting
- FAILURE TO MATCH LINE COUNT WILL CAUSE CRITICAL DATA CORRUPTION

CRITICAL Output Requirements:
- Return EXACTLY the same number of lines as the input (if input has 78 segments, output must have 78 lines)
- Each line in your output corresponds to ONE segment from the input
- Format: Return ONLY the Chinese translation for each segment, one per line
- Do NOT use markdown, code blocks, JSON, or any formatting
- Do NOT add line numbers, explanations, or comments
- Do NOT add blank lines or extra lines
- Do NOT merge or split segments - maintain ONE-TO-ONE correspondence

Segment Boundary Rules (MOST IMPORTANT):
1. NEVER merge multiple segments into one
2. NEVER split one segment into multiple segments
3. Each input segment MUST produce exactly ONE output line
4. Preserve the exact segmentation structure from the input
5. If a segment ends mid-sentence, keep it that way
6. If a segment contains ellipsis (...), dashes, or multiple sentences, DO NOT split it
   - Translate as ONE line without line breaks
   - Example: "But if their house was destroyed... They didn' keep their gold in the house, boy."
   - 可要是连他们的房子全都毁了……他们是不会把黄金放在家里的，孩子！

Translation Rules:
1. Use the reference text to ensure correct translation of:
   - Character names (e.g., "Harry Potter" → "哈利·波特", "Dumbledore" → "邓布利多")
   - Place names (e.g., "Privet Drive" → "女贞路", "Hogwarts" → "霍格沃茨")
   - Magical terms (e.g., "Muggle" → "麻瓜", "wand" → "魔杖")
   - Spell names and magical concepts

2. Match the style and tone of the official translation:
   - Use the same terminology and phrasing where the reference text provides guidance
   - Maintain consistency in character voice and narrative style

3. Handle narration vs. dialogue appropriately:
   - The English text is from an audiobook (narrated version)
   - Translate naturally while preserving the audiobook's narrative style

4. What to preserve:
   - Segment boundaries (never merge or split)
   - One-to-one correspondence with input segments
   - Natural Chinese sentence structure within each segment

Example Input (3 segments):
1. Chapter One. The Boy Who Lived.
2. Mr. and Mrs. Dursley of Number Four Privet Drive were proud to say that they were perfectly normal.
3. They were the last people you'd expect to be involved in anything strange.

Example Output (3 lines, no numbers):
第１章　大难不死的男孩
家住女贞路四号的德思礼夫妇总是得意地说他们是非常规矩的人家。
他们从来跟神秘古怪的事不沾边，因为他们根本不相信那些邪门歪道。

⚠️ SHORT SEGMENT WARNING ⚠️
Even VERY SHORT segments (just 1-3 words) MUST be translated as separate lines!

Example Input with short segments (5 segments):
1. moaned Dobby in a kind of miserable ecstasy.
2. "So noble!
3. So valiant!
4. But he must save himself, he must, Harry Potter must not —"
5. Dobby suddenly froze, his bat ears quivering.

Example Output (5 lines - each short segment gets its own line):
多比既伤心又欢喜地呻吟着。
"多么高贵！
多么勇敢！
但他必须保住自己，他必须，哈利·波特千万不能——"
多比突然僵住了，两只蝙蝠状的耳朵颤抖着。

WRONG Output (merged short segments - DO NOT DO THIS):
多比既伤心又欢喜地呻吟着。"多么高贵！多么勇敢！但他必须保住自己……"
多比突然僵住了。

Important:
- The reference text is for terminology and style guidance
- Output line count MUST equal input segment count
- Maintain segment boundaries exactly as provided
- Use official character/place names from the reference
"""


class TranscriptTranslator:
    """Translates English transcript to Chinese using reference text and LLM."""

    def __init__(self, llm_client: "OpenAIClient", llm_cache_dir: str | None = None):
        """Initialize the translator.

        Args:
            llm_client: OpenAI-compatible client for LLM calls
            llm_cache_dir: Optional directory for per-batch LLM result cache (resume-safe).
        """
        self.llm_client = llm_client
        self.llm_cache_dir = llm_cache_dir

    def _translate_batch(
        self,
        segments: List[TranscriptSegment],
        start_idx: int,
        system_prompt: str,
    ) -> List[str]:
        """Translate a batch of segments.

        Args:
            segments: Batch of segments to translate
            start_idx: Starting index (1-based) for numbering
            system_prompt: System prompt with reference text

        Returns:
            List of translated strings for this batch
        """
        # Format segments as numbered list
        numbered_segments = []
        for i, seg in enumerate(segments, start_idx):
            numbered_segments.append(f"{i}. {seg.text}")

        english_segments_text = "\n".join(numbered_segments)
        batch_size = len(segments)

        # Add explicit count instruction to user input
        user_input = f"""🎯 THIS BATCH: {batch_size} segments → You MUST output EXACTLY {batch_size} lines of Chinese translation!

{english_segments_text}"""

        logger.info(f"Translating batch: segments {start_idx}-{start_idx + batch_size - 1} ({batch_size} segments)")

        model_name = getattr(self.llm_client, "_model_name", "")
        cache_key = (
            TRANSLATION_PROMPT_VERSION,
            model_name,
            start_idx,
            batch_size,
            tuple(seg.text for seg in segments),
            hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
        )
        cached_batch = llm_disk_cache_load(self.llm_cache_dir, "translation", cache_key)
        if cached_batch is not None:
            return cached_batch

        # Retry logic with exponential backoff
        max_retries = 10
        retry_delay = 3  # seconds

        for attempt in range(max_retries):
            try:
                translated_response = self.llm_client.generate(
                    instruction=system_prompt,
                    user_input=user_input,
                )
                if translated_response is None:
                    raise ValueError("LLM returned None response")
                if not translated_response or not translated_response.strip():
                    raise ValueError("LLM returned empty response")
                logger.info(f"Received translated response: {len(translated_response)} characters")
                break  # Success, exit retry loop
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed!")
                    raise  # Re-raise the exception after all retries

        # Parse translated segments (one per line)
        translated_lines = translated_response.strip().split("\n")

        # Filter out empty lines
        translated_segments = [line.strip() for line in translated_lines if line.strip()]

        logger.info(f"Parsed {len(translated_segments)} translated segments from batch")

        # Validation: Check if output count matches input count
        if len(translated_segments) != batch_size:
            error_msg = (
                f"Translation batch segment count mismatch! "
                f"Expected {batch_size} translations but got {len(translated_segments)}. "
                f"Batch range: {start_idx}-{start_idx + batch_size - 1}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        llm_disk_cache_save(self.llm_cache_dir, "translation", cache_key, translated_segments)
        return translated_segments

    def translate(
        self,
        transcript: List[TranscriptSegment],
        reference_text: str,
        batch_size: int = 50,
    ) -> List[str]:
        """Translate English transcript to Chinese using reference text and LLM.

        This function translates segments in batches to improve LLM accuracy in following
        line count requirements. The reference text is placed in the system prompt to
        leverage KV cache across batches.

        Args:
            transcript: English transcript segments to translate
            reference_text: Official Chinese translation reference (e.g., harrypotter_cn.txt chapter)
            batch_size: Number of segments to translate per batch (default: 50)

        Returns:
            List of Chinese translation strings, one per segment (same length as input)

        Example:
            >>> from core.tools.openai_client import OpenAIClient
            >>> llm_client = OpenAIClient()
            >>> translator = TranscriptTranslator(llm_client)
            >>> chinese_translations = translator.translate(
            ...     transcript=english_segments,
            ...     reference_text=chinese_chapter_text,
            ...     batch_size=50,
            ... )
            >>> # chinese_translations[i] corresponds to transcript[i]
        """
        total_segments = len(transcript)
        logger.info(f"Starting batch translation with {total_segments} segments (batch_size={batch_size})")

        # Build system prompt with reference text (will be cached by KV cache)
        system_prompt = f"""{TRANSLATION_PROMPT}

# Chinese Reference Text (for terminology and style guidance)
{reference_text}"""

        # Split into batches and translate
        all_translations = []
        num_batches = (total_segments + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_segments)
            batch_segments = transcript[start:end]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

            # Retry logic for segment count mismatch
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    batch_translations = self._translate_batch(
                        segments=batch_segments,
                        start_idx=start + 1,  # 1-based indexing
                        system_prompt=system_prompt,
                    )
                    break  # Success, exit retry loop
                except ValueError as e:
                    if "segment count mismatch" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Batch translation failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying batch {batch_idx + 1}...")
                        continue
                    else:
                        # Re-raise if it's the last attempt or a different error
                        raise

            all_translations.extend(batch_translations)
            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} complete. "
                f"Total translated: {len(all_translations)}/{total_segments}"
            )

        # Final validation
        if len(all_translations) != total_segments:
            error_msg = (
                f"Final translation count mismatch! "
                f"Expected {total_segments} translations but got {len(all_translations)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Translation complete.")

        return all_translations
