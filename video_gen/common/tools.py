import functools
import hashlib
import json
import math
import os
import pickle
import re
import shlex
import subprocess
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

import pysubs2
import tqdm
import whisperx
from loguru import logger

from video_gen.video_material import TranscriptSegment, TranscriptWord

# Conditional import of decord (not available on macOS ARM)
try:
    import decord
except ImportError:
    decord = None  # type: ignore


def cached(cache_dir: str, exclude_params: Optional[List[str]] = None) -> Callable[..., Any]:
    """Decorator to cache function results to disk using pickle.

    Args:
        cache_dir: Directory to store cache files
        exclude_params: Optional list of parameter names to exclude from cache key generation
                       (blacklist for unpicklable objects like thread locks)

    Returns:
        Decorated function with caching capability

    Example:
        >>> @cached(cache_dir="/tmp/my_cache")
        ... def expensive_function(x, y):
        ...     return x + y

        >>> @cached(cache_dir="/tmp/my_cache", exclude_params=['llm_client'])
        ... def function_with_client(text, llm_client=None):
        ...     return text.upper()
    """
    if exclude_params is None:
        exclude_params = []

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        import inspect

        # Get function signature to map args to parameter names
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        def hash_args(*args, **kwargs) -> str:
            """Hash the args and kwargs to a hex string using pickle serialization.

            Excludes parameters specified in exclude_params from the hash.
            """
            # Filter out excluded parameters from args
            filtered_args = []
            for i, arg in enumerate(args):
                if i < len(param_names) and param_names[i] not in exclude_params:
                    filtered_args.append(arg)
                elif i >= len(param_names):
                    # Extra positional args beyond defined params
                    filtered_args.append(arg)

            # Filter out excluded parameters from kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_params}

            # Use pickle.dumps for reliable serialization of complex objects
            serialized = pickle.dumps((tuple(filtered_args), filtered_kwargs))
            # Use full 64-character hash to minimize collision risk
            return hashlib.sha256(serialized).hexdigest()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            hash_key = hash_args(*args, **kwargs)
            cache_file = os.path.join(cache_dir, f"{hash_key}.pkl")

            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        logger.info(f"Loading cached result from {cache_file}")
                        return pickle.load(f)
                except (pickle.PickleError, EOFError, OSError) as e:
                    logger.warning(f"Failed to load cache from {cache_file}: {e}. Re-computing...")
                    # If cache is corrupted, delete it and continue to recompute
                    try:
                        os.remove(cache_file)
                    except OSError:
                        pass

            # Compute result
            result = func(*args, **kwargs)

            # Try to save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                logger.info(f"Saved result to cache: {cache_file}")
            except (pickle.PickleError, OSError) as e:
                logger.warning(f"Failed to save cache to {cache_file}: {e}")

            return result

        return wrapper

    return decorator


def get_ffmpeg_path() -> Optional[str]:
    """Get path to ffmpeg if available on the current system. First looks at PATH, then checks if
    one is available from the `imageio_ffmpeg` package. Returns None if ffmpeg couldn't be found.
    """
    # Try invoking ffmpeg with the current environment.
    try:
        subprocess.call(["ffmpeg", "-v", "quiet"])
        return "ffmpeg"
    except OSError:
        pass  # Failed to invoke ffmpeg with current environment, try another possibility.

    return None


class CommandTooLong(Exception):
    """Raised if the length of a command line argument exceeds the limit allowed on Windows."""


def invoke_command(args: List[str]) -> int:
    """Same as calling Python's subprocess.call() method, but explicitly
    raises a different exception when the command length is too long.

    See https://github.com/Breakthrough/PySceneDetect/issues/164 for details.

    Arguments:
        args: List of strings to pass to subprocess.call().

    Returns:
        Return code of command.

    Raises:
        CommandTooLong: `args` exceeds built in command line length limit on Windows.
    """
    try:
        return subprocess.call(args)
    except OSError as err:
        if os.name != "nt":
            raise
        exception_string = str(err)
        # Error 206: The filename or extension is too long
        # Error 87:  The parameter is incorrect
        to_match = ("206", "87")
        if any([x in exception_string for x in to_match]):
            raise CommandTooLong() from err
        raise


def split_video_by_shot_boundaries(
    video_path: str,
    shot_boundaries: List[Dict[str, Any]],
    output_dir: str | None = None,
    output_file_template: str = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    ffmpeg_path: str | None = get_ffmpeg_path(),
    show_output: bool = False,
    arg_override: str = "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac",
) -> List[str]:
    """Split video into multiple clips based on shot boundaries using ffmpeg.

    This function takes a video file and splits it into separate clips according to
    provided shot boundaries. Each clip is encoded using ffmpeg with configurable
    encoding parameters.

    Arguments:
        video_path: Path to the input video file.
        shot_boundaries: List of shot boundary dicts with 'start_time' and 'end_time' keys.
                        Time values should be in milliseconds.
                        Example: [{"start_time": 0, "end_time": 5000}, ...]
        output_dir: Optional directory where output files will be saved.
                   If None, files are saved to current directory.
        output_file_template: Template for output filenames with placeholders:
                             $VIDEO_NAME - replaced with input video filename (without extension)
                             $SCENE_NUMBER - replaced with "{index}-{start_time}-{end_time}"
                             Default: "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4"
        ffmpeg_path: Path to ffmpeg executable. If None, uses "ffmpeg" from PATH.
        show_output: If True, shows ffmpeg output for first clip and errors for remaining clips.
                    If False, suppresses all ffmpeg output.
        arg_override: FFmpeg encoding arguments as a string. Supports quoted arguments.
                     Default: "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac"

    Returns:
        List of successfully created output file paths.

    Raises:
        ValueError: If shot_boundaries is empty.

    Example:
        >>> boundaries = [
        ...     {"start_time": 0, "end_time": 5000},      # 0-5 seconds
        ...     {"start_time": 5000, "end_time": 10000},  # 5-10 seconds
        ... ]
        >>> output_files = split_video_by_shot_boundaries(
        ...     "input.mp4",
        ...     boundaries,
        ...     output_dir="clips"
        ... )
        >>> print(output_files)
        ['clips/input-Scene-0-0-5000.mp4', 'clips/input-Scene-1-5000-10000.mp4']
    """

    # Validate input
    if not shot_boundaries:
        raise ValueError("shot_boundaries cannot be empty")

    # Get video metadata using decord
    vr = decord.VideoReader(video_path)
    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
    finally:
        del vr  # Release video reader resources

    # Initialize progress bar
    progress_bar = tqdm.tqdm(total=total_frames, unit="frame", miniters=1, dynamic_ncols=True)

    # Parse ffmpeg encoding arguments (properly handles quoted strings)
    ffmpeg_args = shlex.split(arg_override.replace('\\"', '"'))

    # Pre-calculate video filename (avoid recalculating in loop)
    video_file = os.path.basename(video_path).rsplit(".", 1)[0]

    # Track successfully created output files
    output_files = []

    # Process each shot boundary
    for i, shot_boundary in enumerate(shot_boundaries):
        start_time = shot_boundary["start_time"]
        end_time = shot_boundary["end_time"]
        duration = end_time - start_time

        # Generate output path
        scene_number = f"{i}-{start_time}-{end_time}"
        output_path = output_file_template.replace("$VIDEO_NAME", video_file).replace("$SCENE_NUMBER", scene_number)

        if output_dir:
            output_path = os.path.join(output_dir, output_path)

        # Create parent directory if it exists
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

        # Build ffmpeg command
        call_list = [ffmpeg_path if ffmpeg_path is not None else "ffmpeg"]
        if not show_output:
            call_list += ["-v", "quiet"]
        elif i > 0:
            # Only show ffmpeg output for the first call, which will display any
            # errors if it fails, and then break the loop. We only show error messages
            # for the remaining calls.
            call_list += ["-v", "error"]
        call_list += [
            "-nostdin",
            "-y",
            "-ss",
            str(start_time / 1000),  # Convert milliseconds to seconds
            "-i",
            video_path,
            "-t",
            str(duration / 1000),  # Convert milliseconds to seconds
        ]
        call_list += ffmpeg_args
        call_list += ["-sn"]  # Disable subtitles
        call_list += [str(output_path)]

        logger.info(f"Calling ffmpeg with command: {' '.join(call_list)}")

        # Execute ffmpeg command
        ret_val = invoke_command(call_list)

        if show_output and i == 0 and len(shot_boundaries) > 1:
            logger.info("Output from ffmpeg for Scene 1 shown above, splitting remaining scenes...")

        if ret_val != 0:
            logger.error("Error splitting video (ffmpeg returned %d).", ret_val)
            break

        # Track successful output
        output_files.append(output_path)

        # Update progress bar
        progress_bar.update(math.ceil(duration / 1000 * fps))

    progress_bar.close()

    return output_files


def extract_audio_from_video(
    video_path: str,
    output_path: str,
    audio_format: str = "mp3",
    audio_bitrate: str = "192k",
    sample_rate: int = 44100,
    channels: int = 2,
    ffmpeg_path: str | None = None,
    show_output: bool = False,
) -> int:
    """Extract audio from video file using ffmpeg.

    Arguments:
        video_path: Path to the input video file.
        output_path: Path to save the extracted audio file.
        audio_format: Output audio format (e.g., 'mp3', 'wav', 'aac', 'm4a'). Default is 'mp3'.
        audio_bitrate: Audio bitrate (e.g., '128k', '192k', '320k'). Default is '192k'.
        sample_rate: Audio sample rate in Hz. Default is 44100.
        channels: Number of audio channels (1 for mono, 2 for stereo). Default is 2.
        ffmpeg_path: Path to ffmpeg executable. If None, will auto-detect using get_ffmpeg_path().
        show_output: If True, show ffmpeg output. Default is False.

    Returns:
        Return code from ffmpeg (0 indicates success).

    Raises:
        CommandTooLong: Command line argument exceeds limit on Windows.
        FileNotFoundError: Input video file does not exist.

    Example:
        >>> extract_audio_from_video(
        ...     "video.mp4",
        ...     "audio.mp3",
        ...     audio_format="mp3",
        ...     audio_bitrate="192k"
        ... )
        0
    """
    # Check if input video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get ffmpeg path if not provided
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH.")

    # Build ffmpeg command
    call_list = [ffmpeg_path]

    # Control output verbosity
    if not show_output:
        call_list += ["-v", "quiet"]
    else:
        call_list += ["-v", "info"]

    call_list += [
        "-nostdin",  # Disable interaction
        "-y",  # Overwrite output file without asking
        "-i",
        video_path,  # Input video file
        "-vn",  # Disable video recording
        "-acodec",
        "libmp3lame" if audio_format == "mp3" else "aac" if audio_format in ["aac", "m4a"] else "pcm_s16le",
        "-ab",
        audio_bitrate,  # Audio bitrate
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        str(channels),  # Number of channels
        output_path,  # Output audio file
    ]

    logger.info(f"Extracting audio from video: {video_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Command: {' '.join(call_list)}")

    # Execute ffmpeg command
    ret_val = invoke_command(call_list)

    if ret_val != 0:
        logger.error(f"Failed to extract audio from video. ffmpeg returned {ret_val}")
    else:
        logger.info(f"Successfully extracted audio to: {output_path}")

    return ret_val


def align_transcript_segments(
    transcript_segments: List[TranscriptSegment],
    audio_path: str,
    language: str | None = None,
    return_char_alignments: bool = False,
    device: str = "cpu",
) -> List[TranscriptSegment]:
    """Align transcript segments to get word-level timestamps.

    This function takes transcript segments (from transcribe_with_whisperx with align=False
    or any other source) and performs forced alignment to add word-level timestamps.

    Arguments:
        transcript_segments: List of TranscriptSegment objects to align.
                            Can have empty words lists.
        audio_path: Path to the audio file (needed for alignment).
        language: Language code (e.g., "en", "zh", "es"). If None, will try to detect
                 from transcript content or default to "en".
        return_char_alignments: If True, return character-level alignments as well.
        device: Device to use for alignment ("cpu" or "cuda"). Default is "cpu".

    Returns:
        List of TranscriptSegment objects with word-level timestamps added.
        Original segment timing (start_at, end_at) and text are preserved.

    Raises:
        ImportError: If whisperx is not installed.
        FileNotFoundError: If audio file does not exist.

    Example:
        >>> # First, get basic transcription without alignment (faster)
        >>> segments = transcribe_with_whisperx("audio.mp3", align=False)
        >>>
        >>> # Later, align only the segments you need
        >>> aligned = align_transcript_segments(segments[:10], "audio.mp3", language="en")
        >>> for seg in aligned:
        ...     for word in seg.words:
        ...         print(f"{word.word}: {word.start:.2f}s")
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Auto-detect language if not provided
    if language is None:
        # Try to detect from first segment text (simple heuristic)
        if transcript_segments:
            first_text = transcript_segments[0].text
            # Simple detection: check for Chinese characters
            if any("\u4e00" <= char <= "\u9fff" for char in first_text):
                language = "zh"
            else:
                language = "en"  # Default to English
        else:
            language = "en"

    logger.info(f"Aligning {len(transcript_segments)} segments with language: {language}")

    # Load audio
    audio = whisperx.load_audio(audio_path)

    # Load alignment model
    logger.info("Loading alignment model for word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    # Convert TranscriptSegment to whisperx format
    whisperx_segments = []
    for seg in transcript_segments:
        whisperx_segments.append(
            {
                "start": seg.start_at,
                "end": seg.end_at,
                "text": seg.text,
            }
        )

    # Perform alignment
    logger.info("Aligning transcription to get word-level timestamps...")
    aligned_result = whisperx.align(
        whisperx_segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=return_char_alignments,
    )

    logger.info("Alignment completed successfully.")

    # Convert back to TranscriptSegment with word-level timestamps
    aligned_segments: List[TranscriptSegment] = []
    for i, seg in enumerate(aligned_result["segments"]):
        # Extract words with timestamps
        words = []
        for w in seg.get("words", []):
            # Check if word has all required keys (alignment might fail for some words)
            if "start" not in w or "end" not in w or "score" not in w:
                logger.warning(f"Skipping word without proper alignment: {w.get('word', '<unknown>')}")
                continue
            words.append(
                TranscriptWord(
                    word=w["word"],
                    start=float(w["start"]),
                    end=float(w["end"]),
                    score=float(w["score"]),
                )
            )

        # Create new TranscriptSegment with aligned words
        # Preserve original translations if they exist
        original_seg = transcript_segments[i] if i < len(transcript_segments) else None
        aligned_segment = TranscriptSegment(
            start_at=float(seg["start"]),
            end_at=float(seg["end"]),
            text=seg["text"].strip(),
            trans=original_seg.trans if original_seg else {},
            words=words,
        )
        aligned_segments.append(aligned_segment)

    # Count total words
    total_words = sum(len(seg.words) for seg in aligned_segments)
    logger.info(f"Total words aligned: {total_words}")

    return aligned_segments


@cached(cache_dir="/tmp/cached")
def transcribe_with_whisperx(
    audio_path: str,
    model_name: str = "medium",
    language: str | None = None,
    batch_size: int = 8,
    return_char_alignments: bool = False,
    verbose: bool = False,
    align: bool = True,
) -> List[TranscriptSegment]:
    """Transcribe audio file using WhisperX with optional word-level timestamps.

    This function uses WhisperX to perform automatic speech recognition (ASR).
    By default, it also performs forced alignment to get word-level timestamps.
    You can disable alignment to get faster transcription without word-level timing.

    Arguments:
        audio_path: Path to the audio file to transcribe.
        model_name: Whisper model size. Options: "tiny", "base", "small", "medium", "large-v2".
                   Default is "base" which balances speed and accuracy for CPU.
        language: Language code (e.g., "en", "zh", "es"). If None, will auto-detect.
        batch_size: Batch size for processing. Default is 8 (suitable for CPU).
        return_char_alignments: If True, return character-level alignments as well.
                               Only applies when align=True.
        verbose: If False (default), shows only a progress bar during transcription.
                If True, prints detailed segment information with timestamps.
        align: If True (default), perform word-level forced alignment to get precise
              word timestamps. If False, skip alignment for faster transcription
              (words list will be empty in returned segments).

    Returns:
        List of TranscriptSegment objects containing transcription.
        - If align=True: Each segment includes word-level timestamps with alignment scores.
        - If align=False: Each segment has empty words list, only segment-level timing.

    Raises:
        ImportError: If whisperx is not installed.
        FileNotFoundError: If audio file does not exist.

    Example:
        >>> # With word-level alignment (default)
        >>> segments = transcribe_with_whisperx("audio.mp3", model_name="base")
        >>> for segment in segments:
        ...     print(f"[{segment.start_at:.2f}s - {segment.end_at:.2f}s] {segment.text}")
        ...     for word in segment.words:
        ...         print(f"  {word.word}: {word.start:.2f}s - {word.end:.2f}s")
        >>>
        >>> # Without alignment (faster, no word-level timing)
        >>> segments = transcribe_with_whisperx("audio.mp3", model_name="base", align=False)
        >>> for segment in segments:
        ...     print(f"[{segment.start_at:.2f}s - {segment.end_at:.2f}s] {segment.text}")
    """

    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Configuration for CPU
    device = "cpu"
    compute_type = "int8"  # Best performance for CPU

    logger.info(f"Loading WhisperX model: {model_name} (device={device}, compute_type={compute_type})")

    # Stage 1: Transcribe with whisper model
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    logger.info(f"Transcribing audio: {audio_path}")

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language, verbose=verbose)

    detected_language = result.get("language", "unknown")
    logger.info(f"Detected language: {detected_language}")
    logger.info(f"Transcription completed. Found {len(result['segments'])} segments.")

    # Convert to List[TranscriptSegment] (without word-level timestamps)
    transcript_segments: List[TranscriptSegment] = []
    for seg in result["segments"]:
        transcript_segment = TranscriptSegment(
            start_at=float(seg["start"]),  # Convert np.float64 to Python float
            end_at=float(seg["end"]),  # Convert np.float64 to Python float
            text=seg["text"].strip(),  # Strip whitespace
            trans={},  # Initialize empty translations dict
            words=[],  # Empty list, will be filled by alignment if requested
        )
        transcript_segments.append(transcript_segment)

    logger.info(f"Converted {len(transcript_segments)} segments to TranscriptSegment objects.")

    # Stage 2: Align whisper output for word-level timestamps (optional)
    if align:
        logger.info("Performing word-level alignment...")
        transcript_segments = align_transcript_segments(
            transcript_segments=transcript_segments,
            audio_path=audio_path,
            language=detected_language,
            return_char_alignments=return_char_alignments,
            device=device,
        )
    else:
        logger.info("Skipping word-level alignment (align=False)")

    return transcript_segments


def find_clip_position_in_transcript(
    clip_transcript: List[TranscriptSegment],
    original_transcript: List[TranscriptSegment],
    threshold: float = 0.8,
) -> Tuple[int, int] | None:
    """Find the position where a clip's transcript appears within a longer original transcript.

    This function uses a two-level matching strategy:
    1. Segment-level: Sliding window to find candidate positions in the original
    2. Word-level: One-way fuzzy sequence matching (case-insensitive, ignores punctuation)

    The similarity metric is one-way: "what percentage of clip words are found in the window?"
    This ensures that window size doesn't affect the similarity score - only content matching matters.

    The algorithm tries multiple window sizes (clip_length±few segments) at each position
    to handle ASR variations where segments might be split/merged differently.

    Arguments:
        clip_transcript: Transcript segments from the audio clip to locate.
        original_transcript: Transcript segments from the full original audio.
        threshold: Minimum similarity ratio (0-1) required for a valid match.
                  This represents the minimum percentage of clip words that must be found.
                  Default 0.8 means at least 80% of clip words must be matched.

    Returns:
        Starting index in original_transcript where the clip best matches,
        or None if no match meets the threshold.

    Raises:
        ValueError: If threshold is not in (0, 1].

    Example:
        >>> clip = transcribe_with_whisperx("clip.mp3")
        >>> full = transcribe_with_whisperx("full_episode.mp3")
        >>> position = find_clip_position_in_transcript(clip, full, threshold=0.8)
        >>> if position is not None:
        ...     start_time = full[position].start_at
        ...     print(f"Clip starts at {start_time:.2f}s in the original")

    Note:
        - Uses one-way similarity: matched_words / clip_words (window size doesn't matter)
        - Allows gaps in matching (ASR might miss/add words)
        - Word matching preserves order using SequenceMatcher
        - Example: clip "hello world how are you" in window "hello world how you thanks" = 80% similar
    """

    # Validate parameters
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    def _compute_segment_similarity(
        clip_segments: List[TranscriptSegment],
        original_window: List[TranscriptSegment],
    ) -> float:
        """Compare two segment lists using one-way word-level fuzzy sequence matching.

        Calculates what percentage of clip words are found in the window, using
        SequenceMatcher to allow gaps and preserve order. Window size doesn't affect score.

        Args:
            clip_segments: Segments from the clip to match.
            original_window: Candidate window of segments from the original.

        Returns:
            One-way similarity ratio in [0, 1]: matched_words / clip_words
            1.0 means all clip words are found in the window (perfect match)
            0.5 means half of clip words are found in the window
        """

        def normalize_and_tokenize(text: str) -> List[str]:
            """Normalize text and split into word list."""
            # Remove punctuation and special characters (keep letters, numbers, spaces)
            text = re.sub(r"[^\w\s]", "", text)
            # Convert to lowercase for case-insensitive matching
            text = text.lower()
            # Normalize whitespace: collapse multiple spaces, strip edges
            text = re.sub(r"\s+", " ", text).strip()
            # Split into words
            return text.split()

        def extract_words_from_segments(segments: List[TranscriptSegment]) -> List[str]:
            """Extract and normalize all words from a list of segments."""
            all_words = []
            for segment in segments:
                all_words.extend(normalize_and_tokenize(segment.text))
            return all_words

        # Extract word lists from both segment lists
        clip_words = extract_words_from_segments(clip_segments)
        window_words = extract_words_from_segments(original_window)

        # Handle edge cases
        if not clip_words:
            return 1.0 if not window_words else 0.0
        if not window_words:
            return 0.0

        # Use SequenceMatcher to find matching blocks (preserves word order, allows gaps)
        matcher = SequenceMatcher(None, clip_words, window_words)
        matching_blocks = matcher.get_matching_blocks()

        # Sum up all matched words (excluding the final sentinel block)
        total_matched_words = sum(block.size for block in matching_blocks[:-1])

        # Calculate one-way similarity: what percentage of clip words are in window?
        similarity = total_matched_words / len(clip_words)

        return similarity

    clip_length = len(clip_transcript)
    original_length = len(original_transcript)

    # Edge case 1: Empty clip transcript
    if clip_length == 0:
        logger.warning("clip_transcript is empty, cannot locate segment")
        return None

    # Edge case 2: Clip is longer than original
    if clip_length > original_length:
        logger.warning(
            f"clip_transcript ({clip_length} segments) is longer than "
            f"original_transcript ({original_length} segments), cannot locate"
        )
        return None

    # Two-level sliding window search:
    # Level 1: Try each starting position in the original transcript
    # Level 2: At each start position, try multiple window sizes around clip_length
    #          This handles ASR variations where segments might be split/merged differently

    # Store similarity scores for all [start_position][window_size] combinations
    position_similarity_matrix = []

    # Iterate through all possible starting positions
    for window_start in range(original_length - clip_length + 1):
        # Try different window sizes: clip_length-2 to clip_length+4
        # This symmetric range handles both under-segmentation and over-segmentation by ASR
        end_position_scores = []

        for window_size_offset in range(-2, 5):  # -2, -1, 0, 1, 2, 3, 4
            window_end = window_start + clip_length + window_size_offset

            # Ensure window bounds are valid
            if window_end <= window_start or window_end > original_length:
                continue

            # Extract window from original transcript
            original_window = original_transcript[window_start:window_end]

            # Compute one-way similarity between clip and this window
            similarity = _compute_segment_similarity(clip_transcript, original_window)

            # Store (end_index, similarity_score) tuple
            end_position_scores.append((window_end, similarity))

        position_similarity_matrix.append(end_position_scores)

    # Find the best match across all positions and window sizes
    best_similarity_score = 0.0
    best_start_index = -1
    best_end_index = -1

    for start_idx in range(len(position_similarity_matrix)):
        for end_scores in position_similarity_matrix[start_idx]:
            end_idx, similarity = end_scores
            if similarity > best_similarity_score:
                best_similarity_score = similarity
                best_start_index = start_idx
                best_end_index = end_idx

    # Check if the best match meets the threshold requirement
    if best_similarity_score < threshold:
        logger.warning(
            f"Best match has similarity {best_similarity_score:.1%}, "
            f"below threshold {threshold:.1%}. No valid match found."
        )
        return None

    # Log successful match
    logger.info(
        f"Found match at index {best_start_index} with similarity {best_similarity_score:.1%} "
        f"(window size: {best_end_index - best_start_index} segments)"
    )

    return best_start_index, best_end_index


# Regex patterns for extracting Chinese and English text from ASS subtitle format
# ASS subtitles use tags like {\i1}text{\i0} for styling
CHINESE_PATTERN = r"{\\[^}]*}([^{\\]+?)(?:{\\[^}]*}|$)"
ENGLISH_PATTERN = r"\\N{\\[^}]*}([^{\\]+?)(?:{\\[^}]*}|$)"


def parse_ass(ass_file: str) -> List[TranscriptSegment]:
    r"""Parse ASS subtitle file and convert to TranscriptSegment objects.

    This function extracts Chinese and English text from ASS format subtitles,
    which typically contain style tags like {\i1}text{\i0}. The function handles
    dual-language subtitles where Chinese and English are separated by \N (newline).

    Args:
        ass_file: Path to the ASS subtitle file

    Returns:
        List of TranscriptSegment objects with timing and bilingual text.
        Each segment contains:
        - start_at/end_at: Timing in seconds
        - text: English text (primary)
        - trans: Dictionary with "zh" (Chinese) and "en" (English) translations
        - words: Empty list (no word-level timing from ASS)

    Example:
        >>> segments = parse_ass("episode.ass")
        >>> for seg in segments:
        ...     print(f"[{seg.start_at:.2f}s] {seg.text}")
        ...     print(f"  Chinese: {seg.trans.get('zh', '')}")
    """
    subs = pysubs2.load(ass_file)
    segments: List[TranscriptSegment] = []

    for sub in subs:
        # Extract Chinese text (before \N)
        chinese_match = re.search(CHINESE_PATTERN, sub.text)
        chinese_text = chinese_match.group(1).strip() if chinese_match else ""

        # Extract English text (after \N)
        english_match = re.search(ENGLISH_PATTERN, sub.text)
        english_text = english_match.group(1).strip() if english_match else ""

        # Convert milliseconds to seconds
        start_seconds = sub.start / 1000.0
        end_seconds = sub.end / 1000.0

        # Create TranscriptSegment
        segment = TranscriptSegment(
            start_at=start_seconds,
            end_at=end_seconds,
            text=english_text,  # Use English as primary text
            trans={"zh": chinese_text},  # Bilingual translations
            words=[],  # ASS format doesn't have word-level timing
        )
        segments.append(segment)

    logger.info(f"Parsed {len(segments)} subtitle segments from {ass_file}")
    return segments


def write_ass(
    segments: List[TranscriptSegment],
    output_path: str,
    title: str = "Untitled",
    style_name: str = "Default",
    font_name: str = "Arial",
    font_size: int = 20,
    primary_color: str = "&H00FFFFFF",  # White
    secondary_color: str = "&H00FFFF00",  # Cyan
    outline_color: str = "&H00000000",  # Black
    back_color: str = "&H00000000",  # Black
    bold: bool = False,
    italic: bool = False,
    border_style: int = 1,
    outline: float = 2.0,
    shadow: float = 0.0,
    alignment: int = 2,  # Bottom center
    margin_l: int = 10,
    margin_r: int = 10,
    margin_v: int = 10,
    enable_karaoke: bool = False,
    target_lang_code: str | None = None,
    native_lang_font_name: str | None = None,
    native_lang_font_size_ratio: float = 0.7,
    native_lang_alignment: int = 2,
    native_lang_margin_v: int = 30,
) -> None:
    """Write TranscriptSegment list to ASS subtitle file.

    This function converts a list of TranscriptSegment objects to ASS format
    and writes it to a file. Supports bilingual subtitles (text + translations)
    and optional karaoke-style word-by-word highlighting.

    Arguments:
        segments: List of TranscriptSegment objects to write.
        output_path: Path where the ASS file will be saved.
        title: Video title for ASS metadata. Default is "Untitled".
        style_name: Name of the subtitle style. Default is "Default".
        font_name: Font family name. Default is "Arial".
        font_size: Font size in points. Default is 20.
        primary_color: Primary text color in ASS format (&HAABBGGRR).
                      Default is "&H00FFFFFF" (white).
                      When karaoke is enabled, this is the color of unread words.
        secondary_color: Secondary text color (for karaoke effects).
                        Default is "&H00FFFF00" (cyan/yellow).
                        When karaoke is enabled, this is the highlight color.
        outline_color: Outline/border color. Default is "&H00000000" (black).
        back_color: Shadow color. Default is "&H00000000" (black).
        bold: Whether text should be bold. Default is False.
        italic: Whether text should be italic. Default is False.
        border_style: Border style (1 = outline + shadow, 3 = opaque box).
                     Default is 1.
        outline: Outline thickness in pixels. Default is 2.0.
        shadow: Shadow distance in pixels. Default is 0.0.
        alignment: Text alignment (1-9, numpad layout). Default is 2 (bottom center).
                  Layout: 7 8 9 (top)
                          4 5 6 (middle)
                          1 2 3 (bottom)
        margin_l: Left margin in pixels. Default is 10.
        margin_r: Right margin in pixels. Default is 10.
        margin_v: Vertical margin in pixels. Default is 10.
        enable_karaoke: If True, enable karaoke-style word highlighting using
                       word-level timestamps from segment.words. Words will
                       transition from primary_color to secondary_color as they
                       are spoken. Requires segments to have word-level timing data.
                       Default is False.
        target_lang_code: Language code for the target language (from segment.text).
                         If None, will try to auto-detect from first segment with translation.
                         Examples: "en", "es", "fr", "de", "ja", "ko"
                         Default is None (auto-detect).
        native_lang_font_name: Font name for native language subtitles.
                              If None, uses same font as target language (font_name).
                              Useful for languages requiring specific fonts (e.g., "Microsoft YaHei" for Chinese).
                              Default is None.
        native_lang_font_size_ratio: Ratio of native language font size to target language font size.
                                    Default is 0.7 (70% of target font size).
        native_lang_alignment: Alignment for native language subtitles (1-9, numpad layout).
                              Default is 2 (bottom center).
        native_lang_margin_v: Vertical margin for native language subtitles in pixels.
                             Default is 30.

    Raises:
        ValueError: If segments list is empty.

    Example:
        >>> # Simple usage with English-only subtitles
        >>> segments = [
        ...     TranscriptSegment(
        ...         start_at=0.0,
        ...         end_at=2.5,
        ...         text="Hello world",
        ...         trans={},
        ...         words=[]
        ...     )
        ... ]
        >>> write_ass(segments, "output.ass", title="My Video")
        >>>
        >>> # Bilingual subtitles (English + Chinese)
        >>> segments = [
        ...     TranscriptSegment(
        ...         start_at=0.0,
        ...         end_at=2.5,
        ...         text="Hello world",
        ...         trans={"zh": "你好世界"},
        ...         words=[]
        ...     )
        ... ]
        >>> write_ass(segments, "bilingual.ass", target_lang_code="zh")
        >>>
        >>> # Bilingual subtitles (English + Spanish) with custom native font
        >>> segments = [
        ...     TranscriptSegment(
        ...         start_at=0.0,
        ...         end_at=2.5,
        ...         text="Hello world",
        ...         trans={"es": "Hola mundo"},
        ...         words=[]
        ...     )
        ... ]
        >>> write_ass(segments, "bilingual_es.ass", target_lang_code="es")
        >>>
        >>> # Karaoke-style word highlighting
        >>> segments = [
        ...     TranscriptSegment(
        ...         start_at=0.0,
        ...         end_at=2.5,
        ...         text="Hello world",
        ...         trans={},
        ...         words=[
        ...             TranscriptWord(word="Hello", start=0.0, end=1.0, score=0.98),
        ...             TranscriptWord(word="world", start=1.2, end=2.5, score=0.95),
        ...         ]
        ...     )
        ... ]
        >>> write_ass(segments, "karaoke.ass", enable_karaoke=True,
        ...          primary_color="&H00FFFFFF",    # White (unread)
        ...          secondary_color="&H0000FFFF")  # Yellow (highlighted)

    Note:
        - ASS color format is &HAABBGGRR (hex: Alpha, Blue, Green, Red)
        - If segment has translation for target_lang_code, creates bilingual subtitle with \\N separator
        - Times are automatically converted from seconds to milliseconds
        - Karaoke effect creates separate dialogue lines for each word highlight
        - If enable_karaoke=True but segment.words is empty, falls back to normal subtitle
        - Auto-detection of target_lang_code uses the first translation language found in segments
        - Native language subtitles use a separate style ("NativeLanguage") for customization
    """
    if not segments:
        raise ValueError("segments list cannot be empty")

    # Create ASS file object
    subs = pysubs2.SSAFile()

    # Set metadata
    subs.info["Title"] = title
    subs.info["ScriptType"] = "v4.00+"
    subs.info["WrapStyle"] = "0"  # No automatic wrapping
    subs.info["PlayResX"] = "1920"
    subs.info["PlayResY"] = "1080"
    subs.info["ScaledBorderAndShadow"] = "yes"

    def parse_ass_color(color_str: str) -> pysubs2.Color:
        """Parse ASS color string (&HAABBGGRR) to pysubs2.Color object.

        Args:
            color_str: Color in ASS format like "&H00FFFFFF"

        Returns:
            pysubs2.Color object with RGBA values
        """
        # Remove &H prefix and convert hex to int
        hex_str = color_str.replace("&H", "")
        color_int = int(hex_str, 16)

        # Extract AABBGGRR components
        a = (color_int >> 24) & 0xFF
        b = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        r = color_int & 0xFF

        return pysubs2.Color(r=r, g=g, b=b, a=a)

    # Create style
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = font_size
    style.primarycolor = parse_ass_color(primary_color)
    style.secondarycolor = parse_ass_color(secondary_color)
    style.outlinecolor = parse_ass_color(outline_color)
    style.backcolor = parse_ass_color(back_color)
    style.bold = bold
    style.italic = italic
    style.borderstyle = border_style
    style.outline = outline
    style.shadow = shadow
    style.alignment = alignment
    style.marginl = margin_l
    style.marginr = margin_r
    style.marginv = margin_v

    # Add style to file
    subs.styles[style_name] = style

    # Auto-detect native language code from segments if not provided
    if target_lang_code is None:
        # Try to detect from first segment with translation
        for segment in segments:
            if segment.trans:
                # Get the first translation language code
                target_lang_code = next(iter(segment.trans.keys()))
                logger.info(f"Auto-detected native language code: {target_lang_code}")
                break

    # Helper function to wrap text if too long
    def wrap_text(text: str, max_chars: int = 30) -> str:
        """Wrap text into multiple lines if it exceeds max_chars.

        Args:
            text: Text to wrap
            max_chars: Maximum characters per line (default 30 for 1920px width with 150px margins)

        Returns:
            Text with \\N line breaks inserted at appropriate positions
        """
        if len(text) <= max_chars:
            return text

        # Split into chunks at punctuation or spaces
        result = []
        current_line = ""

        for char in text:
            current_line += char
            # Break at punctuation marks or when reaching max length
            if len(current_line) >= max_chars and char in "，。！？；：,.:;!? ":
                result.append(current_line.strip())
                current_line = ""

        # Add remaining text
        if current_line:
            result.append(current_line.strip())

        # Join with ASS line break
        return "\\N".join(result)

    # Create a second style for native language subtitles (smaller, bottom)
    # Use custom font name if provided, otherwise use same as target language
    native_font = native_lang_font_name if native_lang_font_name else font_name

    native_style = pysubs2.SSAStyle()
    native_style.fontname = native_font
    native_style.fontsize = int(font_size * native_lang_font_size_ratio)
    native_style.primarycolor = parse_ass_color(primary_color)
    native_style.secondarycolor = parse_ass_color(secondary_color)
    native_style.outlinecolor = parse_ass_color(outline_color)
    native_style.backcolor = parse_ass_color(back_color)
    native_style.bold = False
    native_style.italic = False
    native_style.borderstyle = border_style
    native_style.outline = outline
    native_style.shadow = shadow
    native_style.alignment = native_lang_alignment
    native_style.marginl = margin_l  # Use same left margin as target language
    native_style.marginr = margin_r  # Use same right margin as target language
    native_style.marginv = native_lang_margin_v
    subs.styles["NativeLanguage"] = native_style

    # Convert each TranscriptSegment to ASS event
    for segment in segments:
        # Convert seconds to milliseconds
        start_ms = int(segment.start_at * 1000)
        end_ms = int(segment.end_at * 1000)

        # Build subtitle text with optional karaoke effect
        if enable_karaoke and segment.words:
            # Generate karaoke effect using stable-ts approach with continuous display:
            # For each word, create a Dialogue line showing the complete sentence
            # with only that word highlighted in yellow.
            # To prevent flickering, extend each word's display time to the start of next word.
            #
            # Example for "Chapter One":
            #   Word 0 (0.69s-1.21s): {\1c&H00FFFF&}Chapter{\r} One  (延续到下一个词开始)
            #   Word 1 (1.21s-1.35s): Chapter {\1c&H00FFFF&}One{\r}

            # Build word list for the segment
            word_texts = [word.word for word in segment.words]

            # Create one Dialogue line per word (English karaoke in center)
            for word_idx, word in enumerate(segment.words):
                word_start_ms = int(word.start * 1000)

                # Extend display time to next word's start (or segment end if last word)
                if word_idx < len(segment.words) - 1:
                    # Not the last word: display until next word starts
                    word_end_ms = int(segment.words[word_idx + 1].start * 1000)
                else:
                    # Last word: display until segment ends
                    word_end_ms = end_ms

                # Build complete sentence with only current word highlighted
                text_parts = []
                for idx, word_text in enumerate(word_texts):
                    if idx == word_idx:
                        # Current word: yellow color (no bold to avoid alignment issues)
                        text_parts.append(f"{{\\1c&H00FFFF&}}{word_text}{{\\r}}")
                    else:
                        # Other words: keep as-is (default white color)
                        text_parts.append(word_text)

                dialogue_text = " ".join(text_parts)

                # Create event for this word's highlight (English, center)
                event = pysubs2.SSAEvent(
                    start=word_start_ms,
                    end=word_end_ms,
                    text=dialogue_text,
                    style=style_name,
                )
                subs.events.append(event)

            # Also add native language subtitle at bottom (if available)
            if target_lang_code:
                native_text = segment.trans.get(target_lang_code, "")
                if native_text:
                    # Wrap native text if too long
                    wrapped_native = wrap_text(native_text)
                    native_event = pysubs2.SSAEvent(
                        start=start_ms,
                        end=end_ms,
                        text=wrapped_native,
                        style="NativeLanguage",
                    )
                    subs.events.append(native_event)

            # Skip the normal segment event creation below
            continue
        else:
            # Standard subtitle without karaoke
            # If there's a native language translation, create bilingual subtitle
            if target_lang_code:
                native_text = segment.trans.get(target_lang_code, "")
                if native_text:
                    # Bilingual format: NativeLanguage\NTargetLanguage
                    subtitle_text = f"{native_text}\\N{segment.text}"
                else:
                    # Target language only
                    subtitle_text = segment.text
            else:
                # Target language only (no translation configured)
                subtitle_text = segment.text

        # Create SSA event
        event = pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=subtitle_text,
            style=style_name,
        )

        subs.events.append(event)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to file
    subs.save(output_path)
    logger.info(f"Wrote {len(segments)} subtitle segments to {output_path}")

    return None


def extract_m4b_metadata(audio_path: str) -> Dict[str, Any]:
    """Extract complete metadata from m4b audiobook file using ffprobe.

    This function uses ffprobe to extract comprehensive metadata including format info,
    chapters, and stream information from m4b (MPEG-4 audiobook) files.

    Arguments:
        audio_path: Path to the m4b audiobook file.

    Returns:
        Dictionary containing complete metadata with keys:
        - format: Dict with duration, bitrate, format_name, tags (title, artist, etc.)
        - chapters: List of chapter dicts with id, start_time, end_time, tags
        - streams: List of audio stream information

    Raises:
        FileNotFoundError: If audio file does not exist.
        RuntimeError: If ffprobe command fails.

    Example:
        >>> metadata = extract_m4b_metadata("harry_potter.m4b")
        >>> print(f"Title: {metadata['format']['tags'].get('title', 'Unknown')}")
        >>> print(f"Duration: {float(metadata['format']['duration']):.1f}s")
        >>> print(f"Chapters: {len(metadata['chapters'])}")
        >>> for chapter in metadata['chapters']:
        ...     title = chapter.get('tags', {}).get('title', 'Untitled')
        ...     print(f"  - {title}")
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Build ffprobe command
    cmd = [
        "ffprobe",
        "-v",
        "quiet",  # Suppress ffprobe output
        "-print_format",
        "json",  # Output as JSON
        "-show_format",  # Show format/container info
        "-show_chapters",  # Show chapter information
        "-show_streams",  # Show stream information
        audio_path,
    ]

    logger.info(f"Extracting metadata from: {audio_path}")

    # Execute ffprobe command
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed with return code {result.returncode}: {result.stderr}")

    # Parse JSON output
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe JSON output: {e}")

    # Log summary
    num_chapters = len(metadata.get("chapters", []))
    duration = float(metadata.get("format", {}).get("duration", 0))
    logger.info(f"Found {num_chapters} chapters, total duration: {duration:.1f}s")

    return metadata


def parse_m4b_chapters(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse chapter information from ffprobe metadata.

    Extracts chapter data from metadata dictionary and returns a structured list
    with calculated durations and cleaned titles.

    Arguments:
        metadata: Output dictionary from extract_m4b_metadata().

    Returns:
        List of chapter dictionaries, each containing:
        - id: Chapter ID (integer)
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - duration: Duration in seconds (float)
        - title: Chapter title (string, defaults to "Chapter N" if missing)

    Example:
        >>> metadata = extract_m4b_metadata("audiobook.m4b")
        >>> chapters = parse_m4b_chapters(metadata)
        >>> for ch in chapters:
        ...     print(f"Chapter {ch['id']}: {ch['title']} ({ch['duration']:.1f}s)")
        Chapter 0: The Boy Who Lived (1234.5s)
        Chapter 1: The Vanishing Glass (987.6s)
    """
    chapters = []

    for chapter in metadata.get("chapters", []):
        start_time = float(chapter.get("start_time", 0))
        end_time = float(chapter.get("end_time", 0))
        chapter_id = chapter.get("id", len(chapters))

        # Extract title from tags, fallback to default
        title = chapter.get("tags", {}).get("title", f"Chapter {chapter_id}")

        chapters.append(
            {
                "id": chapter_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "title": title,
            }
        )

    logger.info(f"Parsed {len(chapters)} chapters")
    return chapters


def split_m4b_by_chapters(
    audio_path: str,
    chapters: List[Dict[str, Any]],
    output_dir: str,
    audio_format: str = "mp3",
    audio_bitrate: str = "128k",
    sample_rate: int = 44100,
    ffmpeg_path: str | None = None,
    show_output: bool = False,
) -> List[str]:
    """Split m4b audiobook file into separate audio files by chapters.

    This function takes an m4b file and splits it into individual audio files
    according to chapter boundaries. Similar to split_video_by_shot_boundaries
    but specifically designed for audio chapters.

    Arguments:
        audio_path: Path to the input m4b file.
        chapters: List of chapter dicts from parse_m4b_chapters() with keys:
                 'id', 'start_time', 'end_time', 'duration', 'title'.
        output_dir: Directory where output files will be saved.
        audio_format: Output audio format: "mp3", "m4a", "mp4", "wav", "aac".
                     Default is "mp3".
                     Note: For EAC3/Dolby Digital Plus codec, use "mp4" format.
                           M4A format does not support EAC3 codec.
        audio_bitrate: Audio bitrate (e.g., "128k", "192k", "320k").
                      Default is "128k".
                      Special value "copy": Use codec copy mode for zero quality loss
                                            (preserves original codec, bitrate, sample rate).
        sample_rate: Audio sample rate in Hz. Default is 44100.
                    Ignored when audio_bitrate="copy".
        ffmpeg_path: Path to ffmpeg executable. If None, uses get_ffmpeg_path().
        show_output: If True, shows ffmpeg output. Default is False.

    Returns:
        List of successfully created output file paths.

    Raises:
        ValueError: If chapters list is empty.
        RuntimeError: If ffmpeg is not found.
        FileNotFoundError: If input audio file does not exist.

    Example:
        >>> metadata = extract_m4b_metadata("audiobook.m4b")
        >>> chapters = parse_m4b_chapters(metadata)
        >>> # High-quality extraction with re-encoding
        >>> output_files = split_m4b_by_chapters(
        ...     "audiobook.m4b",
        ...     chapters,
        ...     output_dir="chapters",
        ...     audio_format="mp3",
        ...     audio_bitrate="192k"
        ... )
        >>> print(f"Created {len(output_files)} chapter files")
        >>>
        >>> # Zero quality loss with codec copy (fastest, preserves original quality)
        >>> output_files = split_m4b_by_chapters(
        ...     "audiobook.m4b",
        ...     chapters,
        ...     output_dir="chapters",
        ...     audio_format="mp4",     # Use mp4 for EAC3 codec
        ...     audio_bitrate="copy"    # Codec copy mode
        ... )
        >>> print(f"Created {len(output_files)} chapter files with original quality")
    """
    # Validate input
    if not chapters:
        raise ValueError("chapters list cannot be empty")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get ffmpeg path
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH.")

    # Track successfully created files
    output_files = []

    # Initialize progress bar
    progress_bar = tqdm.tqdm(total=len(chapters), unit="chapter", desc="Splitting chapters")

    # Process each chapter
    for chapter in chapters:
        chapter_id = chapter["id"]
        start_time = chapter["start_time"]
        end_time = chapter["end_time"]
        title = chapter["title"]

        # Sanitize title for filename (remove/replace invalid characters)
        safe_title = re.sub(r'[<>:"/\\|?*]', "-", title)
        safe_title = safe_title.strip()

        # Generate output filename
        output_filename = f"chapter_{chapter_id:03d}_{safe_title}.{audio_format}"
        output_path = os.path.join(output_dir, output_filename)

        # Detect codec copy mode
        use_codec_copy = audio_bitrate.lower() == "copy" or audio_bitrate == "0k"

        # Select audio codec based on format or copy mode
        if use_codec_copy:
            audio_codec = "copy"
        elif audio_format == "mp3":
            audio_codec = "libmp3lame"
        elif audio_format in ["aac", "m4a"]:
            audio_codec = "aac"
        elif audio_format == "wav":
            audio_codec = "pcm_s16le"
        else:
            audio_codec = "copy"  # Copy codec if format is unknown

        # Build ffmpeg command
        call_list = [ffmpeg_path]

        # Control output verbosity
        if not show_output:
            call_list += ["-v", "quiet"]
        else:
            call_list += ["-v", "info"]

        call_list += [
            "-nostdin",  # Disable interaction
            "-y",  # Overwrite output file without asking
            "-ss",
            str(start_time),  # Start time in seconds
            "-to",
            str(end_time),  # End time in seconds
            "-i",
            audio_path,  # Input file
            "-vn",  # Disable video (m4b might have cover art)
            "-acodec",
            audio_codec,  # Audio codec
        ]

        # Only add bitrate and sample rate if NOT using codec copy
        # (ffmpeg errors if you specify these with -acodec copy)
        if not use_codec_copy:
            call_list += [
                "-ab",
                audio_bitrate,  # Audio bitrate
                "-ar",
                str(sample_rate),  # Sample rate
            ]

        call_list.append(output_path)  # Output file

        logger.info(f"Splitting chapter {chapter_id}: {title}")

        # Execute ffmpeg command
        ret_val = invoke_command(call_list)

        if ret_val != 0:
            logger.error(f"Failed to split chapter {chapter_id}: {title} (ffmpeg returned {ret_val})")
            continue

        # Track successful output
        output_files.append(output_path)
        logger.info(f"Created: {output_path}")

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()

    logger.info(f"Successfully created {len(output_files)} out of {len(chapters)} chapter files")

    return output_files
