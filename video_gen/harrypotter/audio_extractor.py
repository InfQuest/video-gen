"""Audio extractor for Harry Potter audiobook materials."""

import os
import subprocess

from loguru import logger

from video_gen.common.tools import get_ffmpeg_path
from video_gen.harrypotter.models import M4BChapter


class AudioExtractor:
    """Extracts audio chapters from M4B audiobook files."""

    def __init__(
        self,
        audio_codec: str = "copy",
        audio_bitrate: str | None = None,
        audio_channels: int | None = None,
        sample_rate: int | None = None,
    ):
        """Initialize AudioExtractor with optional audio encoding parameters.

        Args:
            audio_codec: Audio codec to use (default: "copy" for no re-encoding).
                         Options: "copy", "aac", "mp3", "libopus", etc.
            audio_bitrate: Audio bitrate (e.g., "192k", "320k"). Only used when
                          audio_codec is not "copy". Default: None (use encoder default).
            audio_channels: Number of audio channels (1=mono, 2=stereo, 6=5.1).
                           Default: None (keep original).
            sample_rate: Sample rate in Hz (e.g., 44100, 48000).
                        Default: None (keep original).

        Example:
            >>> # Preserve original audio quality (default)
            >>> extractor = AudioExtractor()

            >>> # Convert to stereo AAC at 192k
            >>> extractor = AudioExtractor(
            ...     audio_codec="aac",
            ...     audio_bitrate="192k",
            ...     audio_channels=2
            ... )
        """
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.audio_channels = audio_channels
        self.sample_rate = sample_rate

    def extract_chapter(
        self,
        m4b_path: str,
        chapter: M4BChapter,
        output_path: str,
    ) -> str:
        """Extract a single chapter from M4B audiobook to separate audio file.

        Uses ffmpeg with configurable audio encoding. By default uses codec copy
        for zero quality loss. Can be configured to re-encode audio with different
        codec, bitrate, channels, or sample rate.

        Args:
            m4b_path: Path to the M4B audiobook file
            chapter: Chapter metadata (start_time, end_time, duration)
            output_path: Where to save the extracted audio

        Returns:
            Path to the extracted audio file

        Raises:
            RuntimeError: If ffmpeg is not found or extraction fails
            FileNotFoundError: If m4b_path doesn't exist

        Example:
            >>> # Default: preserve original quality
            >>> extractor = AudioExtractor()
            >>> audio_path = extractor.extract_chapter(
            ...     m4b_path="harrypotter_book1.m4b",
            ...     chapter=chapter_obj,
            ...     output_path="output/chapter1_audio.m4a"
            ... )
            'output/chapter1_audio.m4a'

            >>> # Custom: convert to stereo AAC
            >>> extractor = AudioExtractor(
            ...     audio_codec="aac",
            ...     audio_bitrate="192k",
            ...     audio_channels=2
            ... )
            >>> audio_path = extractor.extract_chapter(...)
        """
        # Verify input file exists
        if not os.path.exists(m4b_path):
            raise FileNotFoundError(f"M4B file not found: {m4b_path}")

        # Check if audio file already exists (caching mechanism)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Audio file already exists, skipping extraction: {output_path}")
            logger.info(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            return output_path

        # Get ffmpeg path
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH.")

        logger.info(f"Extracting chapter {chapter.id}: {chapter.title}")
        logger.info(f"  Duration: {chapter.duration / 60:.2f} minutes")
        logger.info(f"  Output: {output_path}")

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Build ffmpeg command with configurable audio encoding
        cmd = [
            ffmpeg_path,
            "-v",
            "info",  # Show progress
            "-nostdin",  # Disable interaction
            "-y",  # Overwrite output file without asking
            "-ss",
            str(chapter.start_time),  # Start time in seconds
            "-to",
            str(chapter.end_time),  # End time in seconds
            "-i",
            m4b_path,  # Input file
            "-vn",  # Disable video (m4b might have cover art)
            "-acodec",
            self.audio_codec,  # Audio codec (configurable)
        ]

        # Add optional audio encoding parameters (only when not using codec copy)
        if self.audio_codec != "copy":
            if self.audio_bitrate:
                cmd.extend(["-ab", self.audio_bitrate])
            if self.audio_channels:
                cmd.extend(["-ac", str(self.audio_channels)])
            if self.sample_rate:
                cmd.extend(["-ar", str(self.sample_rate)])

        cmd.append(output_path)

        # Execute ffmpeg command
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        # Verify output file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")

        file_size = os.path.getsize(output_path)
        logger.info(f"Successfully extracted chapter audio: {file_size / 1024 / 1024:.2f} MB")

        return output_path
