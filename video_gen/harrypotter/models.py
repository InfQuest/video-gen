"""Data models for Harry Potter audiobook materials."""

from typing import List

from pydantic import BaseModel, Field

from video_gen.video_material import TranscriptSegment


class SceneInfo(BaseModel):
    """Scene information with timing and image data.

    Represents a visual scene in the audiobook with seamless time boundaries.
    Each scene covers a specific time range and has an associated background image.

    Attributes:
        idx: Scene index
        description: Brief description of the scene
        image_url: Path or URL to the scene's background image
        start_time: Scene start time in seconds (seamless with previous scene)
        end_time: Scene end time in seconds (seamless with next scene)
    """

    idx: int
    description: str
    image_url: str | None = None
    start_time: float
    end_time: float


class M4BChapter(BaseModel):
    """M4B audiobook chapter information.

    Represents a single chapter from an m4b audiobook file, including timing
    and metadata information.

    Attributes:
        id: Chapter ID (usually 0-indexed)
        title: Chapter title
        start_time: Start time in seconds
        end_time: End time in seconds
        duration: Duration in seconds
    """

    id: int
    title: str
    start_time: float
    end_time: float
    duration: float


class M4BMetadata(BaseModel):
    """Complete metadata from an m4b audiobook file.

    Contains all relevant information extracted from the m4b container,
    including format details, audio properties, and chapter structure.

    Attributes:
        title: Book title
        artist: Narrator or author name
        album: Album/series name
        duration: Total duration in seconds
        bitrate: Audio bitrate in bits per second
        sample_rate: Audio sample rate in Hz
        chapters: List of chapter information
    """

    title: str = ""
    artist: str = ""
    album: str = ""
    duration: float = 0.0
    bitrate: int = 0
    sample_rate: int = 0
    chapters: List[M4BChapter] = Field(default_factory=list)


class AudiobookChapterMaterial(BaseModel):
    """Learning material generated from audiobook chapter.

    Represents a complete learning unit extracted from a single audiobook chapter,
    including transcription, learning points, scene information, and storage location.

    Attributes:
        chapter: Chapter metadata and timing information
        transcript: ASR transcription with word-level timestamps
        key_points: Learning points extracted by LLM (optional)
        audio_url: S3 URL or local path to the audio file
        scenes: Scene information with timing and images (optional)
        output_files: List of all output file paths generated during processing (local paths)
        cache_files: List of all cache file paths used during processing (local paths)
    """

    chapter: M4BChapter
    transcript: List[TranscriptSegment]
    key_points: List[str] = Field(default_factory=list)
    audio_url: str
    scenes: List[SceneInfo] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    cache_files: List[str] = Field(default_factory=list)
