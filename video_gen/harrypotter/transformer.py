"""Transformer for Harry Potter audiobook materials."""

import os
import re
import tempfile
import traceback
from typing import TYPE_CHECKING

import s3fs
from loguru import logger

from video_gen.core.tools.openai_client import OpenAIClient
from video_gen.common.tools import (
    align_transcript_segments,
    cached,
    extract_m4b_metadata,
    parse_m4b_chapters,
    transcribe_with_whisperx,
)
from video_gen.harrypotter.audio_extractor import AudioExtractor
from video_gen.harrypotter.models import (
    AudiobookChapterMaterial,
    M4BChapter,
    M4BMetadata,
    SceneInfo,
)
from video_gen.harrypotter.packager import OutputPackager
from video_gen.harrypotter.scene_detector import SceneDetector
from video_gen.harrypotter.synthesizer import synthesize_video_from_material
from video_gen.harrypotter.transcript_corrector import TranscriptCorrector
from video_gen.harrypotter.transcript_translator import TranscriptTranslator
from video_gen.video_material import TranscriptSegment

if TYPE_CHECKING:
    from video_gen.core.tools.openai_client import OpenAIClient


class HarryPotterTransformer:
    """Transform Harry Potter audiobook into structured learning materials.

    This transformer processes m4b audiobook files through a complete pipeline:
    1. Extract metadata and chapter information
    2. Split m4b into individual chapter audio files
    3. Transcribe each chapter using WhisperX
    4. Extract learning points using LLM (optional)
    5. Upload to S3 storage
    6. Generate AudiobookMaterial objects

    Example:
        >>> # Process specific chapters
        >>> transformer = HarryPotterTransformer(
        ...     m4b_path="resource/harrypotter.m4b",
        ...     chapter_ids=[1, 2]
        ... )
        >>> materials = transformer.transform(output_dir="output")
        >>> for chapter_id, material in materials.items():
        ...     print(f"Chapter {chapter_id}: {material.chapter.title}")
        ...     print(f"Transcript segments: {len(material.transcript)}")

        >>> # Process all chapters with custom paths
        >>> transformer = HarryPotterTransformer(
        ...     m4b_path="/path/to/audiobook.m4b",
        ...     chapter_ids=None,  # Process all chapters
        ...     english_reference_path="/path/to/english_text.txt",
        ...     chinese_reference_path="/path/to/chinese_text.txt"
        ... )
        >>> materials = transformer.transform(
        ...     output_dir="output",
        ...     synthesize_video=True,
        ...     preview_duration=30.0
        ... )
    """

    def __init__(
        self,
        m4b_path: str,
        chapter_ids: list[int] | None = None,
        english_reference_path: str | None = None,
        chinese_reference_path: str | None = None,
        audio_codec: str = "copy",
        audio_bitrate: str | None = None,
        audio_channels: int | None = None,
        sample_rate: int | None = None,
    ):
        """Initialize transformer with components.

        Args:
            m4b_path: Path to the M4B audiobook file
            chapter_ids: List of chapter IDs to process. If None, processes all chapters.
            english_reference_path: Path to English reference text file (e.g., harrypotter.txt).
                                   If None, uses resource/harrypotter.txt relative to this file.
            chinese_reference_path: Path to Chinese reference text file (e.g., harrypotter_cn.txt).
                                   If None, uses resource/harrypotter_cn.txt relative to this file.
            audio_codec: Audio codec to use for chapter extraction (default: "copy" for no re-encoding).
                        Options: "copy", "aac", "mp3", "libopus", etc.
            audio_bitrate: Audio bitrate (e.g., "192k", "320k"). Only used when audio_codec is not "copy".
            audio_channels: Number of audio channels (1=mono, 2=stereo, 6=5.1). None keeps original.
            sample_rate: Sample rate in Hz (e.g., 44100, 48000). None keeps original.
        """
        self.s3 = s3fs.S3FileSystem()
        self.m4b_path = m4b_path
        self.chapter_ids = chapter_ids

        # Set default reference paths if not provided
        if english_reference_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            english_reference_path = os.path.join(current_dir, "resource", "harrypotter.txt")
        if chinese_reference_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            chinese_reference_path = os.path.join(current_dir, "resource", "harrypotter_cn.txt")

        self.english_reference_path = english_reference_path
        self.chinese_reference_path = chinese_reference_path

        # Initialize components
        self.llm_client = self._get_llm_client()
        self.audio_extractor = AudioExtractor(
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            audio_channels=audio_channels,
            sample_rate=sample_rate,
        )
        self.corrector = TranscriptCorrector(self.llm_client)
        self.translator = TranscriptTranslator(self.llm_client)

    @staticmethod
    def _get_llm_client():
        """Get LLM client from settings or environment.

        Returns:
            OpenAIClient instance

        Raises:
            RuntimeError: If OPENAI_API_KEY is not found
        """
        # Try to get API key from settings first, fallback to environment variable
        try:
            from video_gen.core.configs.config import settings

            api_key = settings.openai_api_key
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in settings or environment variables")

        return OpenAIClient(api_key=api_key, model_name="google/gemini-2.5-pro")

    @staticmethod
    def get_llm_client():
        """Get LLM client (backward compatibility).

        Deprecated: Use _get_llm_client() instead.
        """
        return HarryPotterTransformer._get_llm_client()

    @staticmethod
    def correct_and_align_transcript(
        transcript: list[TranscriptSegment],
        audio_path: str,
        reference_text: str,
        llm_client=None,
    ) -> list[TranscriptSegment]:
        """Correct transcript using reference text and align with word-level timestamps.

        This function:
        1. Corrects the transcript using LLM and reference text
        2. Creates new segments with corrected text
        3. Performs word-level alignment using WhisperX

        Args:
            transcript: Original transcript segments (without or with alignment)
            audio_path: Path to audio file for alignment
            reference_text: Reference text for correction
            llm_client: Optional LLM client. If None, will create one automatically.

        Returns:
            List of corrected and aligned TranscriptSegment objects

        Example:
            >>> transformer = HarryPotterTransformer(chapter_ids=[1])
            >>> # Get initial transcript (fast, no alignment)
            >>> audio_path, transcript = transformer.transform_chapter(
            ...     chapter, transcribe=True, align=False
            ... )
            >>> # Correct and align
            >>> with open("reference.txt") as f:
            ...     reference = f.read()
            >>> aligned = transformer.correct_and_align_transcript(
            ...     transcript, audio_path, reference
            ... )
        """
        if llm_client is None:
            llm_client = HarryPotterTransformer._get_llm_client()

        logger.info("Starting transcript correction with LLM...")
        logger.info(f"  Input segments: {len(transcript)}")
        logger.info(f"  Reference text length: {len(reference_text)} characters")

        # Correct the transcript using corrector
        corrector = TranscriptCorrector(llm_client)
        corrected_texts = corrector.correct(
            transcript=transcript,
            reference_text=reference_text,
            batch_size=10,  # Reduced to avoid connection timeout with large requests
        )

        logger.info("Correction complete!")
        logger.info(f"  Output segments: {len(corrected_texts)}")

        if len(corrected_texts) != len(transcript):
            logger.warning("  ⚠️  Segment count mismatch! This may affect alignment.")

        # Create new transcript segments with corrected text
        corrected_segments = []
        for i, text in enumerate(corrected_texts):
            if i < len(transcript):
                seg = TranscriptSegment(
                    start_at=transcript[i].start_at,
                    end_at=transcript[i].end_at,
                    text=text,
                    trans={},
                    words=[],
                )
                corrected_segments.append(seg)
            else:
                logger.warning(f"  Skipping extra segment {i + 1} from correction")

        # Perform word-level alignment on corrected transcript
        logger.info("Starting word-level alignment on corrected transcript...")

        aligned_segments = align_transcript_segments(
            transcript_segments=corrected_segments,
            audio_path=audio_path,
            language="en",
        )

        logger.info("Alignment complete!")
        logger.info(f"  Aligned segments: {len(aligned_segments)}")
        logger.info(f"  Total words: {sum(len(seg.words) for seg in aligned_segments)}")

        return aligned_segments

    @staticmethod
    def split_long_segments(
        segments: list[TranscriptSegment],
        max_chars: int = 200,
    ) -> list[TranscriptSegment]:
        """Split long transcript segments into smaller ones for better subtitle display.

        Splits segments that exceed max_chars using a multi-level strategy:
        1. First try splitting at sentence boundaries (. ! ?)
        2. Fall back to comma/semicolon boundaries
        3. Final fallback: split by spaces to ensure no segment exceeds max_chars

        Word-level timestamps are redistributed to the new sub-segments.

        Args:
            segments: Input transcript segments (must have word-level timestamps)
            max_chars: Maximum characters before splitting a segment (default: 160)

        Returns:
            List of TranscriptSegment objects, with long segments split into smaller ones

        Example:
            >>> # After alignment
            >>> aligned = transformer.correct_and_align_transcript(...)
            >>> # Split long segments (threshold: 160 chars)
            >>> split = transformer.split_long_segments(aligned, max_chars=160)
            >>> # Continue with translation
            >>> translated = transformer.translate_transcript(split, ...)
        """
        result = []

        for seg in segments:
            text_length = len(seg.text)

            # Short segment - keep as-is
            if text_length <= max_chars:
                result.append(seg)
                continue

            # Long segment - split at sentence boundaries
            logger.debug(f"Splitting long segment ({text_length} chars): {seg.text[:50]}...")

            # Strategy 1: Split by sentence-ending punctuation
            sentence_pattern = r"([^.!?]+[.!?]+\s*)"
            sentences = re.findall(sentence_pattern, seg.text)

            # Strategy 2: Fall back to splitting by commas/semicolons
            if not sentences or len("".join(sentences).strip()) < text_length * 0.8:
                logger.debug("  Falling back to comma/semicolon splitting")
                comma_pattern = r"([^,;]+[,;]\s*)"
                sentences = re.findall(comma_pattern, seg.text)

                # Check if there's trailing text after the last comma
                joined = "".join(sentences)
                if len(joined) < len(seg.text):
                    remainder = seg.text[len(joined) :].strip()
                    if remainder:
                        sentences.append(remainder)

            # If still nothing, use entire text as one sentence
            if not sentences:
                sentences = [seg.text]

            # Strategy 3: Further split any sentence that still exceeds max_chars by spaces
            processed_sentences = []
            for sentence in sentences:
                sentence_stripped = sentence.strip()
                if len(sentence_stripped) <= max_chars:
                    processed_sentences.append(sentence)
                else:
                    # Sentence is still too long, split by spaces
                    logger.debug(f"  Sentence too long ({len(sentence_stripped)} chars), splitting by spaces")
                    words = sentence_stripped.split()
                    current_chunk = ""

                    for word in words:
                        if not current_chunk:
                            current_chunk = word
                        elif len(current_chunk + " " + word) <= max_chars:
                            current_chunk += " " + word
                        else:
                            # Current chunk is full, save it and start new one
                            processed_sentences.append(current_chunk)
                            current_chunk = word

                    # Add final chunk
                    if current_chunk:
                        processed_sentences.append(current_chunk)

            sentences = processed_sentences

            # Build sub-segments by grouping sentences
            current_text = ""
            current_word_indices = []
            word_idx = 0

            for sentence in sentences:
                # Count words in this sentence
                sentence_words = sentence.strip().split()
                sentence_word_count = len(sentence_words)

                # Prepare sentence to add (with space if needed)
                if not current_text:
                    sentence_to_add = sentence
                elif not current_text.endswith(" "):
                    sentence_to_add = " " + sentence
                else:
                    sentence_to_add = sentence

                # Check if adding this sentence would exceed limit
                if current_text and len(current_text + sentence_to_add) > max_chars:
                    # Create a new sub-segment with current accumulated text
                    if current_word_indices and current_word_indices[0] < len(seg.words):
                        start_word = seg.words[current_word_indices[0]]
                        end_word = seg.words[min(current_word_indices[-1], len(seg.words) - 1)]
                        sub_words = [seg.words[i] for i in current_word_indices if i < len(seg.words)]

                        sub_seg = TranscriptSegment(
                            start_at=start_word.start,
                            end_at=end_word.end,
                            text=current_text.strip(),
                            words=sub_words,
                            trans={},  # Translation will be added later
                        )
                        result.append(sub_seg)
                        logger.debug(f"  Created sub-segment: {len(current_text)} chars, {len(sub_words)} words")

                    # Start new sub-segment
                    current_text = sentence
                    current_word_indices = list(range(word_idx, word_idx + sentence_word_count))
                else:
                    # Add to current sub-segment
                    current_text += sentence_to_add
                    current_word_indices.extend(range(word_idx, word_idx + sentence_word_count))

                word_idx += sentence_word_count

            # Add final sub-segment
            if current_text.strip() and current_word_indices and current_word_indices[0] < len(seg.words):
                start_word = seg.words[current_word_indices[0]]
                end_word = seg.words[min(current_word_indices[-1], len(seg.words) - 1)]
                sub_words = [seg.words[i] for i in current_word_indices if i < len(seg.words)]

                sub_seg = TranscriptSegment(
                    start_at=start_word.start,
                    end_at=end_word.end,
                    text=current_text.strip(),
                    words=sub_words,
                    trans={},
                )
                result.append(sub_seg)
                logger.debug(f"  Created final sub-segment: {len(current_text)} chars, {len(sub_words)} words")

        # Log summary
        if len(result) != len(segments):
            logger.info(f"Split long segments: {len(segments)} -> {len(result)} segments")
            split_count = len(result) - len(segments)
            logger.info(f"  {split_count} segments were split")

        return result

    @staticmethod
    def translate_transcript(
        transcript: list[TranscriptSegment],
        reference_text: str,
        llm_client=None,
    ) -> list[TranscriptSegment]:
        """Translate transcript to Chinese using reference text.

        This function translates English transcript segments to Chinese,
        using the official Chinese translation as reference for terminology.

        Args:
            transcript: English transcript segments to translate
            reference_text: Official Chinese translation reference text
            llm_client: Optional LLM client. If None, will create one automatically.

        Returns:
            List of TranscriptSegment objects with Chinese translations in trans["zh"]

        Example:
            >>> transformer = HarryPotterTransformer(chapter_ids=[1])
            >>> # Get English transcript
            >>> material = transformer.transform_chapter(chapter, transcribe=True)
            >>> # Translate
            >>> with open("resource/harrypotter_cn.txt") as f:
            ...     reference = f.read()
            >>> translated = transformer.translate_transcript(
            ...     material.transcript, reference
            ... )
        """
        if llm_client is None:
            llm_client = HarryPotterTransformer._get_llm_client()

        logger.info("Starting transcript translation with LLM...")
        logger.info(f"  Input segments: {len(transcript)}")
        logger.info(f"  Reference text length: {len(reference_text)} characters")

        # Translate the transcript using translator
        translator = TranscriptTranslator(llm_client)
        chinese_translations = translator.translate(
            transcript=transcript,
            reference_text=reference_text,
            batch_size=10,  # Reduced to 10 to match corrector and improve accuracy
        )

        logger.info("Translation complete!")
        logger.info(f"  Output translations: {len(chinese_translations)}")

        # Strict validation: translations must match transcript count exactly
        if len(chinese_translations) != len(transcript):
            error_msg = (
                f"CRITICAL: Translation count mismatch detected in transformer! "
                f"Expected {len(transcript)} translations but got {len(chinese_translations)}. "
                f"This should have been caught by TranscriptTranslator. "
                f"Aborting to prevent misalignment."
            )
            logger.error(error_msg)
            raise AssertionError(error_msg)

        # Add Chinese translations to transcript segments
        translated_segments = []
        for i, seg in enumerate(transcript):
            if i < len(chinese_translations):
                # Create new segment with Chinese translation
                translated_seg = TranscriptSegment(
                    start_at=seg.start_at,
                    end_at=seg.end_at,
                    text=seg.text,
                    trans={"zh": chinese_translations[i]},  # Add Chinese translation
                    words=seg.words,  # Preserve word-level timing
                )
                translated_segments.append(translated_seg)
            else:
                logger.warning(f"  Missing translation for segment {i + 1}")
                # Keep original segment without translation
                translated_segments.append(seg)

        logger.info(f"Added Chinese translations to {len(translated_segments)} segments")

        return translated_segments

    @staticmethod
    @cached(cache_dir="/tmp/cached")
    def extract_text_chapter(input_file: str, chapter_num: int, language: str = "en") -> str | None:
        """Extract a specific chapter from text file.

        Args:
            input_file: Path to input text file (harrypotter.txt or harrypotter_cn.txt)
            chapter_num: Chapter number to extract (1, 2, 3, ...)
            language: Language of the text file ("en" or "zh")

        Returns:
            str: Extracted chapter text, or None if extraction failed

        Example:
            >>> text = HarryPotterTransformer.extract_text_chapter(
            ...     "resource/harrypotter.txt",
            ...     chapter_num=1,
            ...     language="en"
            ... )
            >>> print(f"Chapter 1 length: {len(text)} characters")
        """
        logger.info(f"Extracting chapter {chapter_num} from {input_file} (language: {language})")

        # Try multiple encodings
        if language == "zh":
            encodings = ["utf-8", "gbk", "gb18030"]
        else:
            encodings = ["utf-8", "cp1252", "latin-1"]

        content = None
        for enc in encodings:
            try:
                with open(input_file, encoding=enc) as f:
                    content = f.read()
                logger.info(f"Successfully read file with encoding: {enc}")
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            logger.error(f"Could not read file: {input_file}")
            return None

        # Define chapter patterns based on language
        if language == "zh":
            # Match patterns like "第1章", "第１章", "第一章"
            num_map = {
                1: "[1１一]",
                2: "[2２二]",
                3: "[3３三]",
                4: "[4４四]",
                5: "[5５五]",
                6: "[6６六]",
                7: "[7７七]",
                8: "[8８八]",
                9: "[9９九]",
                10: "(10|１０|十)",  # Use group instead of character class for two-digit numbers
                11: "(11|１１|十一)",
                12: "(12|１２|十二)",
                13: "(13|１３|十三)",
                14: "(14|１４|十四)",
                15: "(15|１５|十五)",
                16: "(16|１６|十六)",
                17: "(17|１７|十七)",
            }
            if chapter_num not in num_map:
                logger.error(f"Chapter number {chapter_num} not supported for Chinese")
                return None

            current_pattern = rf"第{num_map[chapter_num]}章"
            next_pattern = rf"第{num_map.get(chapter_num + 1, '[0-9]+')}章" if chapter_num < 10 else None
        else:
            # English patterns: "CHAPTER ONE" or "Chapter 1"
            num_words = {
                1: "ONE",
                2: "TWO",
                3: "THREE",
                4: "FOUR",
                5: "FIVE",
                6: "SIX",
                7: "SEVEN",
                8: "EIGHT",
                9: "NINE",
                10: "TEN",
                11: "ELEVEN",
                12: "TWELVE",
                13: "THIRTEEN",
                14: "FOURTEEN",
                15: "FIFTEEN",
                16: "SIXTEEN",
                17: "SEVENTEEN",
                18: "EIGHTEEN",
                19: "NINETEEN",
                20: "TWENTY",
            }
            # Support both "CHAPTER ONE" and "Chapter 1" formats
            # Use (?:^|\n) to ensure chapter title is at line start (not in sentence)
            current_pattern = rf"(?:^|\n)\s*CHAPTER\s+(?:{num_words.get(chapter_num, 'XX')}|{chapter_num})\b"
            if chapter_num < 20:
                next_word = num_words.get(chapter_num + 1, "XX")
                next_pattern = rf"(?:^|\n)\s*CHAPTER\s+(?:{next_word}|{chapter_num + 1})\b"
            else:
                next_pattern = None

        # Find chapter positions
        current_match = re.search(current_pattern, content, re.IGNORECASE | re.MULTILINE if language == "en" else re.MULTILINE)
        if not current_match:
            logger.error(f"Could not find pattern: {current_pattern}")
            return None

        # Find next chapter AFTER current chapter (or use end of file)
        start_pos = current_match.start()
        if next_pattern:
            # Search from current chapter position onwards to find the NEXT chapter
            next_match = re.search(next_pattern, content[start_pos + 1:], re.IGNORECASE | re.MULTILINE if language == "en" else re.MULTILINE)
            end_pos = (start_pos + 1 + next_match.start()) if next_match else len(content)
        else:
            end_pos = len(content)

        chapter_content = content[start_pos:end_pos].strip()

        logger.info(f"Extracted chapter {chapter_num}: {len(chapter_content)} characters")
        return chapter_content

    def transform_chapter(
        self,
        chapter: M4BChapter,
        output_dir: str | None = None,
        transcribe: bool = True,
        align: bool = True,
        correct: bool = True,
        split_long_segments: bool = False,
        split_threshold: int = 200,
        translate: bool = True,
        detect_scenes: bool = True,
        generate_scene_images: bool = True,
        synthesize_video: bool = False,
        video_output_path: str | None = None,
        video_width: int = 1920,
        video_height: int = 1080,
        video_fps: int = 24,
        preview_duration: float | None = None,
        reference_text: str | None = None,
        chinese_reference_text: str | None = None,
        auto_extract_reference: bool = True,
    ) -> AudiobookChapterMaterial:
        """Extract audio and optionally transcribe a single chapter from the m4b file.

        This method:
        1. Extracts the audio for a specific chapter using ffmpeg (codec copy)
        2. Optionally transcribes the audio using WhisperX
        3. Optionally extracts reference text from harrypotter.txt
        4. Optionally corrects the transcript using reference text and LLM
        5. Optionally performs word-level alignment on corrected transcript
        6. Optionally translates the transcript to Chinese using reference translation
        7. Optionally detects scenes and generates scene images
        8. Optionally synthesizes video with subtitles and scene backgrounds

        Args:
            chapter: M4BChapter object with timing information
            output_dir: Directory to save the extracted audio file.
                       If None, uses a temporary directory.
            transcribe: If True, transcribe the audio using WhisperX. Default is True.
            align: If True, perform word-level alignment. Default is True.
                   Only used when transcribe=True and correct=False.
            correct: If True, correct the transcript using reference text. Default is True.
                    When True, transcript is first obtained without alignment, then corrected,
                    then aligned for word-level timestamps.
            translate: If True, translate the transcript to Chinese. Default is True.
                      Requires transcribe=True. Uses harrypotter_cn.txt as reference.
            detect_scenes: If True, detect scenes from transcript. Default is True.
                          Requires transcribe=True.
            generate_scene_images: If True, generate images for detected scenes. Default is True.
                                  Requires detect_scenes=True.
            synthesize_video: If True, synthesize video with subtitles and scenes. Default is False.
                             Requires transcript and audio_url to be available.
            video_output_path: Path for output video file. If None, auto-generates based on chapter.
            video_width: Video resolution width in pixels. Default is 1920.
            video_height: Video resolution height in pixels. Default is 1080.
            video_fps: Video frame rate. Default is 24.
            preview_duration: If set, only render first N seconds. None = full video.
            reference_text: English reference text for transcript correction.
                           If None and correct=True and auto_extract_reference=True,
                           will automatically extract from harrypotter.txt.
            chinese_reference_text: Chinese reference text for translation.
                                   If None and translate=True and auto_extract_reference=True,
                                   will automatically extract from harrypotter_cn.txt.
            auto_extract_reference: If True, automatically extract reference texts from
                                   harrypotter.txt and harrypotter_cn.txt when needed.
                                   Default is True.

        Returns:
            AudiobookChapterMaterial: Complete learning material for the chapter

        Example:
            >>> transformer = HarryPotterTransformer(chapter_ids=[1])
            >>> metadata = transformer.transform()
            >>> chapter = metadata.chapters[0]
            >>>
            >>> # Simple transcription with alignment
            >>> material = transformer.transform_chapter(
            ...     chapter, output_dir="output", transcribe=True, align=True
            ... )
            >>>
            >>> # Transcription with correction (auto-extract reference)
            >>> material = transformer.transform_chapter(
            ...     chapter, output_dir="output", transcribe=True, correct=True
            ... )
            >>>
            >>> # Transcription with correction (custom reference)
            >>> with open("custom_reference.txt") as f:
            ...     reference = f.read()
            >>> material = transformer.transform_chapter(
            ...     chapter, output_dir="output", transcribe=True,
            ...     correct=True, reference_text=reference
            ... )
        """
        # Use the m4b path from instance
        m4b_path = self.m4b_path

        # Initialize lists to track generated files
        output_files = []  # All output files generated
        cache_files = []  # All cache files used

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Sanitize chapter title for filename
        safe_title = re.sub(r'[<>:"/\\|?*]', "-", chapter.title)
        safe_title = safe_title.strip()

        # Generate output filename (use mp4 format for EAC3 codec compatibility)
        output_filename = f"chapter_{chapter.id:03d}_{safe_title}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Extract audio using audio extractor
        output_path = self.audio_extractor.extract_chapter(
            m4b_path=m4b_path,
            chapter=chapter,
            output_path=output_path,
        )
        output_files.append(output_path)  # Track audio file

        # Load or transcribe audio
        transcript = None
        material_json_path = os.path.join(output_dir, f"chapter{chapter.id}_material.json")

        # Try to load existing material first
        if not transcribe and os.path.exists(material_json_path):
            logger.info(f"Loading existing material from: {material_json_path}")
            with open(material_json_path, encoding="utf-8") as f:
                existing_material = AudiobookChapterMaterial.model_validate_json(f.read())
                transcript = existing_material.transcript
                logger.info(f"  Loaded {len(transcript)} transcript segments")

        # Transcribe audio if requested
        if transcribe:
            logger.info("Starting ASR transcription with WhisperX...")
            logger.info("  Model: medium")
            logger.info("  Language: en")
            logger.info(f"  This may take several minutes for a {chapter.duration / 60:.1f} minute audio...")

            # If correction is requested, transcribe without alignment first
            should_align = align and not correct

            transcript = transcribe_with_whisperx(
                audio_path=output_path,
                model_name="medium",  # Good balance of speed and accuracy
                language="en",  # Harry Potter is in English
                verbose=True,  # Show progress
                align=should_align,  # Skip alignment if we're going to correct
            )

            logger.info("Transcription complete!")
            logger.info(f"  Total segments: {len(transcript)}")
            if should_align:
                logger.info(f"  Total words: {sum(len(seg.words) for seg in transcript)}")

            # Correct transcript if requested
            if correct:
                # Auto-extract reference text if not provided
                if reference_text is None:
                    if auto_extract_reference:
                        logger.info("Auto-extracting reference text from configured English reference path...")
                        english_input = self.english_reference_path

                        if not os.path.exists(english_input):
                            raise FileNotFoundError(
                                f"Reference text file not found: {english_input}. "
                                "Please provide reference_text parameter or set auto_extract_reference=False."
                            )

                        reference_text = HarryPotterTransformer.extract_text_chapter(
                            input_file=english_input, chapter_num=chapter.id, language="en"
                        )

                        if not reference_text:
                            raise ValueError(
                                f"Failed to extract chapter {chapter.id} from {english_input}. "
                                "Please provide reference_text parameter manually."
                            )

                        logger.info(f"  Extracted reference text: {len(reference_text)} characters")
                    else:
                        raise ValueError(
                            "reference_text is required when correct=True and auto_extract_reference=False"
                        )

                # Use the static method to correct and align
                transcript = HarryPotterTransformer.correct_and_align_transcript(
                    transcript=transcript,
                    audio_path=output_path,
                    reference_text=reference_text,
                )

        # Split long segments if requested (after alignment, before translation)
        if split_long_segments and transcript:
            logger.info("Splitting long segments...")
            original_count = len(transcript)
            transcript = HarryPotterTransformer.split_long_segments(
                segments=transcript,
                max_chars=split_threshold,
            )
            if len(transcript) != original_count:
                logger.info(f"  Segments after split: {len(transcript)}")

        # Translate transcript if requested (moved outside transcribe block)
        if translate and transcript:
            # Auto-extract Chinese reference text if not provided
            if chinese_reference_text is None:
                if auto_extract_reference:
                    logger.info("Auto-extracting Chinese reference text from configured Chinese reference path...")
                    chinese_input = self.chinese_reference_path

                    assert os.path.exists(chinese_input), (
                        f"Chinese reference text file not found: {chinese_input}. "
                        "Provide chinese_reference_text parameter or set translate=False."
                    )

                    chinese_reference_text = HarryPotterTransformer.extract_text_chapter(
                        input_file=chinese_input, chapter_num=chapter.id, language="zh"
                    )

                    assert chinese_reference_text, (
                        f"Failed to extract chapter {chapter.id} from {chinese_input}. "
                        "Check if the chapter exists in the Chinese reference file."
                    )

                    logger.info(f"  Extracted Chinese reference text: {len(chinese_reference_text)} characters")
                else:
                    raise AssertionError(
                        "chinese_reference_text is required when translate=True and auto_extract_reference=False. "
                        "Provide chinese_reference_text parameter or set translate=False."
                    )

            # Perform translation with reference text
            assert chinese_reference_text, "Chinese reference text is required for translation"
            transcript = HarryPotterTransformer.translate_transcript(
                transcript=transcript,
                reference_text=chinese_reference_text,
            )
            logger.info("Transcript translation completed!")

        # Detect scenes and generate images if requested
        scenes = []
        if detect_scenes and transcript:
            logger.info("Starting scene detection...")
            scene_detector = SceneDetector()

            # Detect scenes from transcript
            scene_result = scene_detector.detect_scenes(transcript)
            logger.info(f"Detected {len(scene_result.scenes)} unique scenes")
            logger.info(f"Generated {len(scene_result.scene_segments)} scene-segment mappings")

            # Generate scene images if requested
            if generate_scene_images:
                logger.info("Generating scene images...")
                scene_images_dir = os.path.join(output_dir, f"chapter_{chapter.id}_scenes")
                scene_result = scene_detector.generate_scene_images(
                    result=scene_result,
                    output_dir=scene_images_dir,
                    size="2560x1440",  # Force specific resolution instead of "2K"
                )
                logger.info(f"Generated {sum(1 for s in scene_result.scenes if s.image_url)} scene images")

                # Track all generated scene images
                for scene in scene_result.scenes:
                    if scene.image_url and os.path.exists(scene.image_url):
                        output_files.append(scene.image_url)

            # Convert SceneDetectionResult to List[SceneInfo] with seamless time boundaries
            # We need to create continuous time segments from scene_segments
            for i, scene_seg in enumerate(scene_result.scene_segments):
                # Find the scene definition
                scene = next((s for s in scene_result.scenes if s.idx == scene_seg.scene_idx), None)
                if scene:
                    # Use scene_seg times which are already derived from transcript
                    scene_info = SceneInfo(
                        idx=i,  # Use sequential index for seamless ordering
                        description=scene.description,
                        image_url=scene.image_url,
                        start_time=scene_seg.start_time,
                        end_time=scene_seg.end_time,
                    )
                    scenes.append(scene_info)

            logger.info(f"Created {len(scenes)} scene info objects with seamless boundaries")

        # Save material JSON file first
        # We'll update it later with output_files and cache_files
        with open(material_json_path, "w", encoding="utf-8") as f:
            # Temporary material without output_files
            temp_material = AudiobookChapterMaterial(
                chapter=chapter,
                transcript=transcript or [],
                key_points=[],
                audio_url=output_path,
                scenes=scenes,
            )
            f.write(temp_material.model_dump_json(indent=2))
        output_files.append(material_json_path)  # Track material JSON file
        logger.info(f"Saved material to: {material_json_path}")

        # Synthesize video if requested
        if synthesize_video:
            # Generate video output path if not provided
            if video_output_path is None:
                if preview_duration:
                    video_filename = f"chapter{chapter.id}_preview.mp4"
                else:
                    video_filename = f"chapter{chapter.id}_full.mp4"
                video_output_path = os.path.join(output_dir, video_filename)

            logger.info("=" * 80)
            logger.info("Starting Video Synthesis")
            logger.info("=" * 80)
            logger.info(f"  Material: {material_json_path}")
            logger.info(f"  Output: {video_output_path}")
            logger.info(f"  Resolution: {video_width}x{video_height}")
            logger.info(f"  FPS: {video_fps}")
            if preview_duration:
                logger.info(f"  Preview Duration: {preview_duration}s")
            else:
                logger.info(f"  Duration: Full chapter ({chapter.duration:.2f}s)")

            synthesize_video_from_material(
                material_json_path=material_json_path,
                output_path=video_output_path,
                video_width=video_width,
                video_height=video_height,
                fps=video_fps,
                enable_karaoke=True,
                preview_duration=preview_duration,
            )

            logger.info("=" * 80)
            logger.info("✅ Video Synthesis Complete!")
            logger.info(f"Output: {video_output_path}")
            logger.info("=" * 80)

            output_files.append(video_output_path)  # Track video file

        # TODO: Track cache files from WhisperX, corrector, translator
        # For now, we can scan the cache directory for recently modified files
        cache_dir = "/tmp/cached"
        if os.path.exists(cache_dir):
            current_time = os.path.getmtime(output_path)  # Use audio file time as reference
            # Include cache files modified around the same time (within 1 hour)
            for cache_file in os.listdir(cache_dir):
                if cache_file.endswith(".pkl"):
                    cache_path = os.path.join(cache_dir, cache_file)
                    if abs(os.path.getmtime(cache_path) - current_time) < 3600:  # Within 1 hour
                        cache_files.append(cache_path)

        # Create final AudiobookChapterMaterial with all tracked files
        material = AudiobookChapterMaterial(
            chapter=chapter,
            transcript=transcript or [],
            key_points=[],
            audio_url=output_path,
            scenes=scenes,
            output_files=output_files,
            cache_files=cache_files,
        )

        # Update material JSON file with complete information
        with open(material_json_path, "w", encoding="utf-8") as f:
            f.write(material.model_dump_json(indent=2, exclude_none=True))
        logger.info(f"Updated material with {len(output_files)} output files and {len(cache_files)} cache files")

        return material

    def transform(
        self,
        output_dir: str | None = None,
        transcribe: bool = True,
        correct: bool = True,
        align: bool = True,
        split_long_segments: bool = False,
        split_threshold: int = 200,
        translate: bool = True,
        detect_scenes: bool = True,
        generate_scene_images: bool = True,
        synthesize_video: bool = False,
        video_width: int = 1920,
        video_height: int = 1080,
        video_fps: int = 24,
        preview_duration: float | None = None,
    ) -> dict[int, AudiobookChapterMaterial]:
        """Transform Harry Potter audiobook chapters.

        This method:
        1. Extracts metadata and chapter information
        2. Processes each chapter (audio extraction, transcription, correction, alignment, translation)
        3. Optionally synthesizes video with subtitles and scene backgrounds
        4. Returns results for all chapters

        Args:
            output_dir: Directory to save output files. If None, uses ./output
            transcribe: If True, transcribe audio. Default is True.
            correct: If True, correct transcript with reference text. Default is True.
            align: If True, perform word-level alignment. Default is True.
            split_long_segments: If True, split long segments for better subtitle display. Default is False.
            split_threshold: Maximum characters per segment before splitting. Default is 160.
            translate: If True, translate transcript to Chinese. Default is True.
            detect_scenes: If True, detect scenes from transcript. Default is True.
            generate_scene_images: If True, generate images for detected scenes. Default is True.
            synthesize_video: If True, synthesize video for each chapter. Default is False.
            video_width: Video resolution width in pixels. Default is 1920.
            video_height: Video resolution height in pixels. Default is 1080.
            video_fps: Video frame rate. Default is 24.
            preview_duration: If set, only render first N seconds. None = full video.

        Returns:
            dict: Mapping of chapter_id to AudiobookChapterMaterial

        Example:
            >>> transformer = HarryPotterTransformer(chapter_ids=[1, 2])
            >>> results = transformer.transform(output_dir="output")
            >>> for chapter_id, material in results.items():
            ...     print(f"Chapter {chapter_id}: {len(material.transcript)} segments")

            >>> # With video synthesis
            >>> results = transformer.transform(
            ...     output_dir="output",
            ...     synthesize_video=True,
            ...     video_width=1920,
            ...     video_height=1080
            ... )
        """
        # Get metadata
        metadata = self._extract_metadata()

        # Set default output directory if not provided
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "output")

        logger.info(f"Processing {len(metadata.chapters)} chapters...")

        results = {}
        for chapter in metadata.chapters:
            logger.info(f"Processing chapter {chapter.id}: {chapter.title}")

            try:
                material = self.transform_chapter(
                    chapter=chapter,
                    output_dir=output_dir,
                    transcribe=transcribe,
                    correct=correct,
                    align=align,
                    split_long_segments=split_long_segments,
                    split_threshold=split_threshold,
                    translate=translate,
                    detect_scenes=detect_scenes,
                    generate_scene_images=generate_scene_images,
                    synthesize_video=synthesize_video,
                    video_width=video_width,
                    video_height=video_height,
                    video_fps=video_fps,
                    preview_duration=preview_duration,
                )

                results[chapter.id] = material
                logger.info(f"  ✓ Completed chapter {chapter.id}")

            except Exception as e:
                logger.error(f"  ✗ Failed to process chapter {chapter.id}: {e}")

                traceback.print_exc()

        logger.info(f"Processed {len(results)}/{len(metadata.chapters)} chapters successfully")

        return results

    def _extract_metadata(self) -> M4BMetadata:
        """Extract metadata from Harry Potter m4b audiobook file.

        Returns:
            M4BMetadata: Complete audiobook metadata including filtered chapters
        """
        # Extract raw metadata using ffprobe
        raw_metadata = extract_m4b_metadata(self.m4b_path)

        # Parse chapters from raw metadata
        parsed_chapters = parse_m4b_chapters(raw_metadata)

        # If no chapters found in metadata, treat entire file as one chapter
        if not parsed_chapters:
            format_info = raw_metadata.get("format", {})
            format_tags = format_info.get("tags", {})
            duration = float(format_info.get("duration", 0.0))
            title = format_tags.get("title", "Chapter 1")

            # Use first chapter_id if specified, otherwise default to 1
            chapter_id = self.chapter_ids[0] if self.chapter_ids else 1

            logger.info(f"No chapter metadata found, treating entire file as chapter {chapter_id}")
            parsed_chapters = [
                {
                    "id": chapter_id,
                    "title": title,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                }
            ]

        # Filter chapters if self.chapters is specified
        if self.chapter_ids:
            filtered_chapters = [ch for ch in parsed_chapters if ch["id"] in self.chapter_ids]
        else:
            filtered_chapters = parsed_chapters

        # Convert chapter dicts to M4BChapter objects
        chapter_objects = [
            M4BChapter(
                id=ch["id"],
                title=ch["title"],
                start_time=ch["start_time"],
                end_time=ch["end_time"],
                duration=ch["duration"],
            )
            for ch in filtered_chapters
        ]

        # Extract format metadata
        format_info = raw_metadata.get("format", {})
        format_tags = format_info.get("tags", {})

        # Get audio stream info (bitrate, sample_rate)
        streams = raw_metadata.get("streams", [])
        audio_stream = streams[0] if streams else {}

        # Construct M4BMetadata object
        metadata = M4BMetadata(
            title=format_tags.get("title", ""),
            artist=format_tags.get("artist", ""),
            album=format_tags.get("album", ""),
            duration=float(format_info.get("duration", 0.0)),
            bitrate=int(format_info.get("bit_rate", 0)),
            sample_rate=int(audio_stream.get("sample_rate", 0)),
            chapters=chapter_objects,
        )

        return metadata

    def package_outputs(
        self,
        output_dir: str,
        results: dict[int, AudiobookChapterMaterial],
        cache_dir: str = "/tmp/cached",
        include_cache: bool = True,
    ) -> str:
        """Package all output files (videos, images, materials, caches) into a zip archive.

        This method delegates to OutputPackager for the actual packaging logic.

        Args:
            output_dir: The output directory containing generated files
            results: Dictionary mapping chapter_id to AudiobookChapterMaterial
            cache_dir: Path to cache directory. Default is /tmp/cached
            include_cache: Whether to include cache files. Default is True

        Returns:
            str: Path to the created zip file

        Example:
            >>> transformer = HarryPotterTransformer(m4b_path="...", chapter_ids=[1])
            >>> results = transformer.transform(output_dir="output")
            >>> zip_path = transformer.package_outputs(output_dir="output", results=results)
            >>> print(f"Package created: {zip_path}")
        """
        packager = OutputPackager()
        return packager.package(
            output_dir=output_dir,
            results=results,
            cache_dir=cache_dir,
            include_cache=include_cache,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Harry Potter Audiobook Transformer - 处理 M4B 音频书生成视频学习材料")

    # Input files
    parser.add_argument(
        "--m4b-path",
        type=str,
        default=None,
        help="Path to M4B audiobook file (default: resource/harrypotter.m4b)",
    )
    parser.add_argument(
        "--english-reference",
        type=str,
        default=None,
        help="Path to English reference text (default: resource/harrypotter.txt)",
    )
    parser.add_argument(
        "--chinese-reference",
        type=str,
        default=None,
        help="Path to Chinese reference text (default: resource/harrypotter_cn.txt)",
    )

    # Chapter selection
    parser.add_argument(
        "--chapters",
        type=str,
        default="1",
        help='Comma-separated chapter IDs (e.g., "1,2,3") or "all" for all chapters (default: "1")',
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./output)",
    )

    # Processing flags
    parser.add_argument("--no-transcribe", action="store_true", help="Skip ASR transcription")
    parser.add_argument("--no-correct", action="store_true", help="Skip transcript correction")
    parser.add_argument("--no-align", action="store_true", help="Skip word-level alignment")
    parser.add_argument("--no-translate", action="store_true", help="Skip Chinese translation")
    parser.add_argument("--no-scenes", action="store_true", help="Skip scene detection")
    parser.add_argument("--no-scene-images", action="store_true", help="Skip scene image generation")
    parser.add_argument("--no-video", action="store_true", help="Skip video synthesis")

    # Segment splitting options
    parser.add_argument(
        "--split-long-segments",
        action="store_true",
        help="Split long segments for better subtitle display (default: False)",
    )
    parser.add_argument(
        "--split-threshold",
        type=int,
        default=200,
        help="Maximum characters per segment before splitting (default: 200)",
    )

    # Audio encoding options
    parser.add_argument(
        "--audio-codec",
        type=str,
        default="copy",
        help='Audio codec for chapter extraction (default: "copy" for no re-encoding). '
        "Options: copy, aac, mp3, libopus",
    )
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default=None,
        help="Audio bitrate (e.g., 192k, 320k). Only used when --audio-codec is not copy",
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=None,
        help="Number of audio channels (1=mono, 2=stereo, 6=5.1). Default: keep original",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate in Hz (e.g., 44100, 48000). Default: keep original",
    )

    # Video options
    parser.add_argument("--video-width", type=int, default=1920, help="Video width (default: 1920)")
    parser.add_argument("--video-height", type=int, default=1080, help="Video height (default: 1080)")
    parser.add_argument("--video-fps", type=int, default=24, help="Video FPS (default: 24)")
    parser.add_argument(
        "--preview-duration",
        type=float,
        default=None,
        help="Preview duration in seconds (default: None = full video)",
    )

    # Packaging options
    parser.add_argument(
        "--package",
        action="store_true",
        help="Package all outputs (videos, images, materials, caches) into a zip file after processing",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/cached",
        help="Cache directory path (default: /tmp/cached)",
    )
    parser.add_argument(
        "--no-cache-in-package",
        action="store_true",
        help="Exclude cache files from the package",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Harry Potter Audiobook Transformer")
    print("=" * 80)

    # Determine M4B path
    if args.m4b_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # If single chapter specified, try chapter{n}.m4b first, fallback to harrypotter.m4b
        if args.chapters.lower() != "all" and "," not in args.chapters:
            chapter_num = args.chapters.strip()
            chapter_file = os.path.join(base_dir, "resource", f"chapter{chapter_num}.m4b")
            if os.path.exists(chapter_file):
                args.m4b_path = chapter_file
            else:
                args.m4b_path = os.path.join(base_dir, "resource", "harrypotter.m4b")
        else:
            args.m4b_path = os.path.join(base_dir, "resource", "harrypotter.m4b")

    # Parse chapter IDs
    if args.chapters.lower() == "all":
        chapter_ids = None
    else:
        chapter_ids = [int(ch.strip()) for ch in args.chapters.split(",")]

    # Determine output directory
    if args.output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(base_dir, "output")

    print("\n配置:")
    print(f"  M4B 文件: {args.m4b_path}")
    print(f"  章节: {chapter_ids if chapter_ids else 'All chapters'}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  音频编码: {args.audio_codec}", end="")
    if args.audio_codec != "copy":
        details = []
        if args.audio_bitrate:
            details.append(f"{args.audio_bitrate}")
        if args.audio_channels:
            details.append(f"{args.audio_channels}ch")
        if args.sample_rate:
            details.append(f"{args.sample_rate}Hz")
        if details:
            print(f" ({', '.join(details)})")
        else:
            print()
    else:
        print(" (保持原始质量)")
    print(f"  视频尺寸: {args.video_width}x{args.video_height} @ {args.video_fps}fps")
    if args.preview_duration:
        print(f"  预览模式: {args.preview_duration}秒")
    print()

    # Create transformer
    transformer = HarryPotterTransformer(
        m4b_path=args.m4b_path,
        chapter_ids=chapter_ids,
        english_reference_path=args.english_reference,
        chinese_reference_path=args.chinese_reference,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        audio_channels=args.audio_channels,
        sample_rate=args.sample_rate,
    )

    # Transform with specified options
    results = transformer.transform(
        output_dir=args.output_dir,
        transcribe=not args.no_transcribe,
        correct=not args.no_correct,
        align=not args.no_align,
        split_long_segments=args.split_long_segments,
        split_threshold=args.split_threshold,
        translate=not args.no_translate,
        detect_scenes=not args.no_scenes,
        generate_scene_images=not args.no_scene_images,
        synthesize_video=not args.no_video,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        preview_duration=args.preview_duration,
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"✅ 处理完成! 共处理 {len(results)} 章")
    print("=" * 80)

    for chapter_id, material in results.items():
        print(f"\n章节 {chapter_id}: {material.chapter.title}")
        print(f"  音频: {material.audio_url}")
        print(f"  转录段数: {len(material.transcript)}")
        print(f"  场景数: {len(material.scenes)}")

        # Save material JSON
        output_file = os.path.join(args.output_dir, f"chapter{chapter_id}_material.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(material.model_dump_json(indent=2, exclude_none=True))
        print(f"  材料文件: {output_file}")

    print(f"\n所有输出文件保存在: {args.output_dir}")

    # Package outputs if requested
    if args.package:
        print("\n打包输出文件...")
        try:
            zip_path = transformer.package_outputs(
                output_dir=args.output_dir,
                results=results,
                cache_dir=args.cache_dir,
                include_cache=not args.no_cache_in_package,
            )
            print(f"\n📦 打包完成: {zip_path}")
        except Exception as e:
            print(f"\n⚠️  打包失败: {e}")
            traceback.print_exc()
