from typing import Dict, List

from pydantic import BaseModel


class TranscriptWord(BaseModel):
    """Represents a single word in a transcript with precise timing and alignment score.

    Attributes:
        word: The transcribed word text
        start: Start time of the word in seconds
        end: End time of the word in seconds
        score: Alignment confidence score (0-1) from forced alignment model.
              Higher scores indicate more reliable word-level timestamps.
    """

    word: str
    start: float
    end: float
    score: float


class TranscriptSegment(BaseModel):
    """Represents a single segment of video subtitles with timing information.

    Attributes:
        start_at: Start time of the segment in seconds
        end_at: End time of the segment in seconds
        text: Original text content of the segment
        trans: Translations of the text in different languages (language code -> translated text)
        words: List of word-level timestamps with alignment scores for precise timing
    """

    start_at: float
    end_at: float
    text: str
    trans: Dict[str, str]
    words: List[TranscriptWord]


class VideoMaterial(BaseModel):
    """Complete video material with subtitles, explanations, and key points.

    Attributes:
        video_url: URL of the video file
        transcript: List of subtitle segments for the main video
        explanation_audio_url: URL of the explanation audio file
        explanation_transcript: List of subtitle segments for the explanation audio
        key_points: List of key learning points from the video
        explanation_play_time: Optimal time (in seconds) to play the explanation audio.
            This is typically after the related dialogue scene completes.
            None indicates the time point could not be determined.
    """

    video_url: str
    transcript: List[TranscriptSegment]
    explanation_audio_url: str
    explanation_transcript: List[TranscriptSegment]
    key_points: List[str]
    explanation_play_time: float | None = None
