"""Transcript corrector for Harry Potter audiobook materials."""

import re
import time
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from video_gen.common.tools import cached
from video_gen.video_material import TranscriptSegment

if TYPE_CHECKING:
    from video_gen.core.tools.openai_client import OpenAIClient


# Correction prompt template
CORRECTION_PROMPT = """You are an expert ASR transcript corrector.

You will receive:
1. ASR Transcript Segments: A numbered list of segments from automatic speech recognition (may contain errors)
2. Reference Text: The authoritative source text (book chapter)

Your task: Correct EACH segment based on the reference text while preserving segment boundaries.

🚨 CRITICAL LINE COUNT REQUIREMENT 🚨
- Your output MUST have EXACTLY the same number of lines as the input segments
- If input has 52 segments, output MUST have EXACTLY 52 lines (not 51, not 53, EXACTLY 52)
- Each line in your output corresponds to ONE segment from the input
- Count your output lines carefully before submitting
- FAILURE TO MATCH LINE COUNT WILL CAUSE CRITICAL DATA CORRUPTION

CRITICAL Output Requirements:
- Return EXACTLY the same number of segments as the input (if input has 78 segments, output must have 78 segments)
- Each line in your output corresponds to ONE segment from the input
- Format: Return ONLY the corrected text for each segment, one per line
- Do NOT use markdown, code blocks, JSON, or any formatting
- Do NOT add line numbers, explanations, or comments
- Do NOT add blank lines or extra lines
- Do NOT merge or split segments - maintain ONE-TO-ONE correspondence

Segment Boundary Rules (MOST IMPORTANT):
1. NEVER merge multiple segments into one
2. NEVER split one segment into multiple segments
3. Each input segment MUST produce exactly ONE output line
4. Preserve the exact segmentation structure from the input
5. If a segment ends mid-sentence, keep it that way (alignment will handle this)

Correction Rules:
1. Correct obvious ASR errors within each segment:
   - Spelling errors (e.g., "blond" → "blonde")
   - Homophones (e.g., "their" vs "there", "too" vs "to")
   - Mumbled words or misheard words
   - Capitalization (proper nouns like "Potters", "Muggle", etc.)

2. CRITICAL - Preserve ALL ASR content exactly (only fix word errors):
   - The ASR is from an AUDIOBOOK read aloud by a narrator
   - The narrator may speak dialogue tags, narration, and all book content
   - DO NOT remove ANY content from ASR, even if it differs from reference
   - DO NOT add content that's not in ASR
   - ONLY fix word-level errors: wrong words, spelling, capitalization
   - When uncertain, keep ASR EXACTLY as transcribed
   - Only fix ERRORS, never "improve" or "align to reference"

3. What NOT to change:
   - DO NOT add words that are not in the ASR (especially dialogue tags)
   - DO NOT remove words that are in the ASR
   - DO NOT change word order
   - DO NOT split sentences (keep "Harvey or Harold" as is, don't make it "Harvey. Or Harold.")
   - DO NOT merge sentences
   - DO NOT add punctuation that changes meaning (like turning statement into question)
   - DO NOT remove or change numbers - keep ALL numbers exactly as they appear in ASR
     (whether written as digits "1, 2, 3" or words "one, two, three")
   - DO NOT convert number formats (e.g., keep "Chapter 1" as "Chapter 1", not "Chapter One")

4. What you CAN change:
   - Fix spelling errors
   - Fix capitalization
   - Fix punctuation (commas, periods) to match correct grammar
   - Fix obvious word substitutions (like "porters" → "Potters")

Example Input (3 segments):
1. Chapter Won. The Boy Who Lived.
2. Mr. and Mrs. Dursley of Number For Privet Drive were proud too say that they were perfectly normal.
3. They were the last people you'd expect too be involved in anything strange.

Example Output (3 lines, no numbers):
Chapter One. The Boy Who Lived.
Mr. and Mrs. Dursley of Number Four Privet Drive were proud to say that they were perfectly normal.
They were the last people you'd expect to be involved in anything strange.

⚠️ SHORT SEGMENT WARNING ⚠️
Even VERY SHORT segments (just 1-3 words) MUST be corrected as separate lines!

Example Input with short segments (5 segments):
1. moaned Doby in a kind of miserable ecstasy.
2. "So noble!
3. So valiant!
4. But he must save himself, he must, Harry Porter must not —"
5. Doby suddenly froze, his bat ears quivering.

Example Output (5 lines - each short segment gets its own line):
moaned Dobby in a kind of miserable ecstasy.
"So noble!
So valiant!
But he must save himself, he must, Harry Potter must not —"
Dobby suddenly froze, his bat ears quivering.

WRONG Output (merged short segments - DO NOT DO THIS):
moaned Dobby in a kind of miserable ecstasy. "So noble! So valiant! But he must..."
Dobby suddenly froze.

WRONG Examples (DO NOT DO THIS):
❌ ASR: "No. Why?"
   Reference: "No," she said sharply. "Why?"
   WRONG: "No," she said sharply. "Why?"  ← Added "she said sharply" from reference!
   CORRECT: No. Why?  ← Keep ASR as-is, audiobook narrator doesn't say "she said"

❌ ASR: "I think they'd be a bit more careful."
   Reference: "You'd think they'd be a bit more careful."
   WRONG: "You'd think they'd be a bit more careful."  ← Changed pronoun!
   CORRECT: I think they'd be a bit more careful.  ← Keep ASR pronoun

❌ ASR: "It might have been Harvey or Harold."
   Reference: "It might have been Harvey. Or Harold."
   WRONG: It might have been Harvey. Or Harold.  ← Split sentence!
   CORRECT: It might have been Harvey or Harold.  ← Keep as one sentence

❌ ASR: "Chapter 1. The Worst Birthday."
   Reference: "Chapter One. The Worst Birthday."
   WRONG: Chapter One. The Worst Birthday.  ← Changed number format!
   CORRECT: Chapter 1. The Worst Birthday.  ← Keep "1" as digit, don't convert to "One"

❌ ASR: "We should all be in position at 8 o'clock."
   Reference: "We should all be in position at eight o'clock."
   WRONG: We should all be in position at eight o'clock.  ← Changed number format!
   CORRECT: We should all be in position at 8 o'clock.  ← Keep "8" as digit

Important:
- The reference text is for correction guidance only
- Do NOT try to match the reference text structure - follow ASR structure
- Segment boundaries are sacred - never change them
- Output line count MUST equal input segment count
"""


class TranscriptCorrector:
    """Corrects ASR transcript errors using reference text and LLM."""

    def __init__(self, llm_client: "OpenAIClient"):
        """Initialize the corrector.

        Args:
            llm_client: OpenAI-compatible client for LLM calls
        """
        self.llm_client = llm_client

    def _find_best_match_position(self, search_text: str, reference_text: str) -> tuple[int, float]:
        """Find the best match position for search_text in reference_text using fuzzy matching.

        Args:
            search_text: Text to search for
            reference_text: Text to search in

        Returns:
            Tuple of (position, match_ratio)
        """
        search_text = re.sub(r"\s+", " ", search_text).strip()

        if not search_text:
            return 0, 0.0

        best_ratio = 0.0
        best_position = 0

        # Slide a window across the reference text
        window_len = len(search_text)
        step_size = max(50, window_len // 4)

        for i in range(0, len(reference_text) - window_len + 1, step_size):
            ref_window = reference_text[i : i + window_len]
            ratio = SequenceMatcher(None, search_text.lower(), ref_window.lower()).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_position = i

        return best_position, best_ratio

    def _find_reference_window(
        self,
        segments: List[TranscriptSegment],
        reference_text: str,
        token_padding: int = 100,
    ) -> Optional[str]:
        """Find and extract a relevant window from the reference text for the given segments.

        Uses the first 3 segments to find the start position and last 3 segments to find
        the end position, then extracts the window with padding.

        Args:
            segments: List of transcript segments to find in reference
            reference_text: Full reference text
            token_padding: Number of tokens (~4 chars each) to include before/after (default: 100)

        Returns:
            Extracted window of reference text, or full text if matching fails
        """
        if len(segments) == 0:
            return reference_text

        # 1. Use first 3 segments to find start position
        start_segments = segments[: min(3, len(segments))]
        start_text = " ".join([seg.text for seg in start_segments])

        logger.info(f"Finding start position with first {len(start_segments)} segments: {start_text[:80]}...")
        start_pos, start_ratio = self._find_best_match_position(start_text, reference_text)
        logger.info(f"  Start match at position {start_pos} (ratio: {start_ratio:.2f})")

        # 2. Use last 3 segments to find end position
        end_segments = segments[-min(3, len(segments)) :]
        end_text = " ".join([seg.text for seg in end_segments])

        logger.info(f"Finding end position with last {len(end_segments)} segments: {end_text[:80]}...")
        end_pos, end_ratio = self._find_best_match_position(end_text, reference_text)
        logger.info(f"  End match at position {end_pos} (ratio: {end_ratio:.2f})")

        # 3. Validate positions
        if start_ratio < 0.3 or end_ratio < 0.3:
            logger.warning(
                f"Low match ratio (start: {start_ratio:.2f}, end: {end_ratio:.2f}), "
                f"using full reference text as fallback"
            )
            return reference_text

        if end_pos < start_pos:
            logger.warning(
                f"End position ({end_pos}) before start position ({start_pos}), using full reference text as fallback"
            )
            return reference_text

        # 4. Extract window with padding
        # Approximate: 1 token ≈ 4 characters in English
        char_padding = token_padding * 4

        window_start = max(0, start_pos - char_padding)
        window_end = min(len(reference_text), end_pos + len(end_text) + char_padding)

        extracted_window = reference_text[window_start:window_end]

        logger.info(
            f"Extracted reference window: {len(extracted_window)} chars "
            f"(from position {window_start} to {window_end} in {len(reference_text)} char reference)"
        )

        return extracted_window

    def _correct_batch(
        self,
        segments: List[TranscriptSegment],
        start_idx: int,
        reference_text: str,
        token_padding: int = 100,
    ) -> List[str]:
        """Correct a batch of segments.

        Args:
            segments: Batch of segments to correct
            start_idx: Starting index (1-based) for numbering
            reference_text: Full reference text (will be truncated to relevant window)
            token_padding: Number of tokens to pad before/after matched positions (default: 100)

        Returns:
            List of corrected strings for this batch
        """
        # Extract relevant window from reference text
        reference_window = self._find_reference_window(
            segments=segments, reference_text=reference_text, token_padding=token_padding
        )

        # Build system prompt with truncated reference
        system_prompt = f"""{CORRECTION_PROMPT}

# Reference Text (for correction guidance)
{reference_window}"""

        # Format segments as numbered list
        numbered_segments = []
        for i, seg in enumerate(segments, start_idx):
            numbered_segments.append(f"{i}. {seg.text}")

        asr_segments_text = "\n".join(numbered_segments)
        batch_size = len(segments)

        # Add explicit count instruction to user input
        user_input = f"""🎯 THIS BATCH: {batch_size} segments → You MUST output EXACTLY {batch_size} lines!

{asr_segments_text}"""

        logger.info(f"Correcting batch: segments {start_idx}-{start_idx + batch_size - 1} ({batch_size} segments)")

        # Retry logic with exponential backoff
        max_retries = 10
        retry_delay = 3  # seconds

        for attempt in range(max_retries):
            try:
                corrected_response = self.llm_client.generate(
                    instruction=system_prompt,
                    user_input=user_input,
                )
                if corrected_response is None:
                    raise ValueError("LLM returned None response")
                if not corrected_response or not corrected_response.strip():
                    raise ValueError("LLM returned empty response")
                logger.info(f"Received corrected response: {len(corrected_response)} characters")

                # Parse corrected segments (one per line)
                corrected_lines = corrected_response.strip().split("\n")

                # Filter out empty lines
                corrected_segments = [line.strip() for line in corrected_lines if line.strip()]

                logger.info(f"Parsed {len(corrected_segments)} corrected segments from batch")

                # Validation: Check if output count matches input count
                if len(corrected_segments) != batch_size:
                    raise ValueError(
                        f"Segment count mismatch: expected {batch_size}, got {len(corrected_segments)}"
                    )

                return corrected_segments

            except Exception as e:
                logger.error(f"Batch correction failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed!")
                    raise  # Re-raise the exception after all retries

        # Should not reach here, but just in case
        raise RuntimeError("Unexpected: exited retry loop without returning or raising")

    @cached(cache_dir="/tmp/cached", exclude_params=["self"])
    def correct(
        self,
        transcript: List[TranscriptSegment],
        reference_text: str,
        batch_size: int = 25,
        token_padding: int = 100,
    ) -> List[str]:
        """Correct ASR transcript using reference text and LLM.

        This function corrects segments in batches. For each batch, it extracts a relevant
        window from the reference text by matching the first and last 3 segments of the batch
        to find the corresponding start and end positions in the reference text.

        Args:
            transcript: Original ASR transcript segments
            reference_text: Reference text for correction (e.g., book chapter)
            batch_size: Number of segments to correct per batch (default: 25)
            token_padding: Number of tokens (~4 chars) to pad before/after (default: 100)

        Returns:
            List of corrected text strings, one per segment (same length as input)

        Example:
            >>> from core.tools.openai_client import OpenAIClient
            >>> llm_client = OpenAIClient()
            >>> corrector = TranscriptCorrector(llm_client)
            >>> corrected_segments = corrector.correct(
            ...     transcript=asr_segments,
            ...     reference_text=chapter_text,
            ...     batch_size=25,
            ...     token_padding=100,
            ... )
            >>> # corrected_segments[i] corresponds to transcript[i]
            >>> # Later: use WhisperX alignment on each corrected segment
        """
        total_segments = len(transcript)
        logger.info(
            f"Starting batch correction with {total_segments} segments "
            f"(batch_size={batch_size}, token_padding={token_padding})"
        )
        logger.info(f"Full reference text size: {len(reference_text)} characters")

        # Split into batches and correct
        all_corrections = []
        num_batches = (total_segments + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_segments)
            batch_segments = transcript[start:end]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

            batch_corrections = self._correct_batch(
                segments=batch_segments,
                start_idx=start + 1,  # 1-based indexing
                reference_text=reference_text,
                token_padding=token_padding,
            )

            all_corrections.extend(batch_corrections)
            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} complete. "
                f"Total corrected: {len(all_corrections)}/{total_segments}"
            )

        # Final validation
        if len(all_corrections) != total_segments:
            error_msg = (
                f"Final correction count mismatch! "
                f"Expected {total_segments} corrections but got {len(all_corrections)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Correction complete.")

        return all_corrections
