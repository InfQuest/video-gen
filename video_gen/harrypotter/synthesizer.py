"""Video synthesizer for Harry Potter audiobook materials.

This module creates videos with synchronized text subtitles from audiobook materials.
Uses ffmpeg + ASS subtitles for fast, high-quality rendering.
"""

import os
import subprocess
import tempfile

from loguru import logger

from video_gen.common.tools import write_ass
from video_gen.harrypotter.models import AudiobookChapterMaterial


def synthesize_video_from_material(
    material_json_path: str,
    output_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: int = 24,
    enable_karaoke: bool = True,
    preview_duration: float | None = None,
    font_name: str = "Helvetica",
) -> str:
    """Synthesize video with synchronized subtitles from audiobook material.

    This function creates a video using ffmpeg + ASS subtitles for fast rendering:
    1. Load material data and validate audio
    2. Generate ASS subtitle file with optional karaoke effect
    3. Use ffmpeg to render black background + audio + subtitles

    Args:
        material_json_path: Path to AudiobookChapterMaterial JSON file
        output_path: Where to save the output video
        video_width: Output video width in pixels
        video_height: Output video height in pixels
        fps: Frames per second for output video
        enable_karaoke: Enable word-level highlighting effect
        preview_duration: If set, only render the first N seconds (for testing).
                         None means render the full video.
        font_name: Font family name for subtitles. Default is "Arial".
                  Common options: Arial, Helvetica, Times New Roman, Verdana,
                  Microsoft YaHei (for Chinese), SimHei, Impact, etc.

    Returns:
        Path to the generated video file

    Raises:
        FileNotFoundError: If material_json_path or audio file doesn't exist
        ValueError: If material data is invalid
        subprocess.CalledProcessError: If ffmpeg fails

    Example:
        >>> # Full video
        >>> synthesize_video_from_material(
        ...     material_json_path="output/chapter1_material.json",
        ...     output_path="output/chapter1_video.mp4",
        ...     enable_karaoke=True
        ... )
        'output/chapter1_video.mp4'

        >>> # Preview (first 30 seconds)
        >>> synthesize_video_from_material(
        ...     material_json_path="output/chapter1_material.json",
        ...     output_path="output/chapter1_preview.mp4",
        ...     enable_karaoke=True,
        ...     preview_duration=30.0
        ... )
        'output/chapter1_preview.mp4'

        >>> # Custom font
        >>> synthesize_video_from_material(
        ...     material_json_path="output/chapter1_material.json",
        ...     output_path="output/chapter1_custom_font.mp4",
        ...     font_name="Microsoft YaHei"
        ... )
        'output/chapter1_custom_font.mp4'
    """
    logger.info("=" * 80)
    logger.info("Starting Video Synthesis (ffmpeg + ASS)")
    if preview_duration:
        logger.info(f"Preview Mode: First {preview_duration} seconds")
    logger.info("=" * 80)

    # 1. Load material data
    logger.info(f"Loading material from: {material_json_path}")
    if not os.path.exists(material_json_path):
        raise FileNotFoundError(f"Material file not found: {material_json_path}")

    with open(material_json_path, encoding="utf-8") as f:
        material = AudiobookChapterMaterial.model_validate_json(f.read())

    logger.info(f"Chapter: {material.chapter.title}")
    logger.info(f"Total segments: {len(material.transcript)}")

    # Filter segments and scenes if preview mode is enabled
    if preview_duration:
        original_count = len(material.transcript)
        material.transcript = [seg for seg in material.transcript if seg.start_at < preview_duration]
        logger.info(f"Preview mode: filtered to {len(material.transcript)} segments (from {original_count} total)")

        # Also filter scenes to only include those within the preview duration
        if material.scenes:
            original_scene_count = len(material.scenes)
            material.scenes = [scene for scene in material.scenes if scene.start_time < preview_duration]
            # Adjust the end_time of the last scene if it extends beyond preview_duration
            if material.scenes and material.scenes[-1].end_time > preview_duration:
                material.scenes[-1].end_time = preview_duration
            logger.info(f"Preview mode: filtered to {len(material.scenes)} scenes (from {original_scene_count} total)")

    # 2. Validate audio file
    audio_path = material.audio_url
    logger.info(f"Audio file: {audio_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get audio duration using ffprobe
    try:
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        total_duration = float(result.stdout.strip())
        logger.info(f"Audio duration: {total_duration:.2f}s ({total_duration / 60:.1f} minutes)")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get audio duration: {e.stderr}")
        raise

    # 3. Generate ASS subtitle file
    logger.info(f"Generating ASS subtitles (karaoke: {enable_karaoke})...")

    # Create temporary ASS file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False, encoding="utf-8") as ass_file:
        ass_path = ass_file.name

        # Add mock Chinese translations for testing
        for segment in material.transcript:
            if not segment.trans.get("zh"):
                # Mock translation: just add a prefix
                segment.trans["zh"] = f"[中文翻译] {segment.text[:30]}..."

    try:
        write_ass(
            segments=material.transcript,
            output_path=ass_path,
            enable_karaoke=enable_karaoke,
            font_name=font_name,  # Use custom font
            primary_color="&H00FFFFFF",  # White text
            secondary_color="&H00000000",  # Black (not used in new karaoke mode)
            outline_color="&H00000000",  # Black outline
            back_color="&H00000000",  # Black shadow
            font_size=80,  # Larger font for English
            bold=False,  # Normal text by default, bold added per-word in karaoke
            outline=2.0,  # Outline width
            shadow=0.0,  # No shadow
            alignment=5,  # Center middle (numpad 5) for English
            margin_l=100,  # Left margin to keep distance from screen edge
            margin_r=100,  # Right margin to keep distance from screen edge
            margin_v=50,
        )
        logger.info(f"  Subtitle file created: {ass_path}")

        # 4. Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 5. Render video with ffmpeg
        logger.info("Rendering video with ffmpeg...")
        logger.info(f"  Resolution: {video_width}x{video_height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info("  This should be much faster than MoviePy (10-20x speedup)...")

        # Escape ASS file path for ffmpeg subtitles filter
        # On Windows, need to escape backslashes and colons
        ass_path_escaped = ass_path.replace("\\", "\\\\\\\\").replace(":", "\\\\:")

        # Determine actual video duration
        render_duration = preview_duration if preview_duration else total_duration

        # Build ffmpeg command with scene images if available
        ffmpeg_cmd = ["ffmpeg", "-y"]  # Overwrite output file

        # Check if we have scene images to use as background
        # Helper function to resolve relative paths
        def resolve_path(path: str) -> str:
            """Resolve relative path based on material_json_path location."""
            if os.path.isabs(path):
                return path
            # If path exists as-is, use it
            if os.path.exists(path):
                return path
            # Get the harrypotter directory (parent of output directory)
            material_dir = os.path.dirname(os.path.abspath(material_json_path))
            harrypotter_dir = os.path.dirname(material_dir)  # Go up from output/
            # Try removing the leading src/materials/videos/harrypotter/ prefix if present
            if path.startswith("src/materials/videos/harrypotter/"):
                rel_path = path.replace("src/materials/videos/harrypotter/", "")
                full_path = os.path.join(harrypotter_dir, rel_path)
                if os.path.exists(full_path):
                    return full_path
            # Try joining directly with harrypotter dir
            full_path = os.path.join(harrypotter_dir, path)
            if os.path.exists(full_path):
                return full_path
            return path

        has_scenes = bool(material.scenes and any(s.image_url for s in material.scenes))

        if has_scenes:
            logger.info(f"Using {len(material.scenes)} scene images as backgrounds")

            # Add all scene images as inputs
            for scene in material.scenes:
                if scene.image_url:
                    resolved_path = resolve_path(scene.image_url)
                    logger.debug(
                        f"Scene image: {scene.image_url} -> {resolved_path} (exists: {os.path.exists(resolved_path)})"
                    )
                    if os.path.exists(resolved_path):
                        ffmpeg_cmd.extend(["-loop", "1", "-i", resolved_path])
                        # Update scene with resolved path for later use
                        scene.image_url = resolved_path
                    else:
                        logger.warning(f"Scene image not found: {resolved_path}")

            # Add audio as last input
            ffmpeg_cmd.extend(["-i", audio_path])

            # Build filter_complex to overlay and transition scenes
            filter_parts = []
            audio_input_idx = len(material.scenes)  # Audio is after all scene images

            # Scale and crop each scene image to fit video dimensions
            for i, scene in enumerate(material.scenes):
                if scene.image_url and os.path.exists(scene.image_url):
                    filter_parts.append(
                        f"[{i}:v]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
                        f"crop={video_width}:{video_height},setsar=1[img{i}]"
                    )

            # Create time-based scene transitions using select and concat
            # We'll use the 'select' filter to show each image during its time range
            concat_inputs = []
            for i, scene in enumerate(material.scenes):
                if scene.image_url and os.path.exists(scene.image_url):
                    # Calculate scene duration
                    scene_duration = scene.end_time - scene.start_time

                    # Trim the image to the scene duration and set PTS
                    filter_parts.append(f"[img{i}]trim=duration={scene_duration},setpts=PTS-STARTPTS[scene{i}]")
                    concat_inputs.append(f"[scene{i}]")

            # Concatenate all scenes (or use single scene directly if only one)
            if len(concat_inputs) > 1:
                filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=1:a=0[bg_raw]")
            elif len(concat_inputs) == 1:
                # If only one scene, use it directly without concat
                filter_parts.append(f"{concat_inputs[0]}copy[bg_raw]")
            else:
                raise ValueError("No valid scene images found for video rendering")

            # Add 70% semi-transparent black overlay for better subtitle readability
            # Create a black color source with 70% opacity (alpha=0.7)
            filter_parts.append(f"color=c=black@0.7:s={video_width}x{video_height}:d={render_duration}[overlay]")

            # Blend the overlay with the background
            filter_parts.append("[bg_raw][overlay]overlay[bg]")

            # Apply subtitles on top of the background with overlay
            filter_parts.append(f"[bg]subtitles={ass_path_escaped}[v]")

            filter_complex = ";".join(filter_parts)

            # Add duration limit for audio in preview mode
            audio_filter = []
            if preview_duration:
                audio_filter = ["-t", str(preview_duration)]

            ffmpeg_cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v]",
                    "-map",
                    f"{audio_input_idx}:a:0",  # Explicitly map only first audio stream
                ]
                + audio_filter
                + [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "faster",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",  # Copy audio without re-encoding (preserves 6-channel Dolby quality)
                    "-shortest",  # End when shortest stream ends
                    output_path,
                ]
            )
        else:
            # Fallback to black background if no scenes
            logger.info("No scene images found, using black background")
            ffmpeg_cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={video_width}x{video_height}:d={render_duration}",
                    "-i",
                    audio_path,
                ]
            )

            # Add duration limit for audio in preview mode
            if preview_duration:
                ffmpeg_cmd.extend(["-t", str(preview_duration)])

            ffmpeg_cmd.extend(
                [
                    "-filter_complex",
                    f"[0:v]subtitles={ass_path_escaped}[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "1:a:0",  # Explicitly map only first audio stream
                    "-c:v",
                    "libx264",
                    "-preset",
                    "faster",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",  # Copy audio without re-encoding (preserves 6-channel Dolby quality)
                    output_path,
                ]
            )

        logger.info("  Running: ffmpeg ...")

        # Run ffmpeg
        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise immediately
        )

        if process.returncode != 0:
            logger.error("ffmpeg failed:")
            logger.error(f"  stdout: {process.stdout}")
            logger.error(f"  stderr: {process.stderr}")
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd, process.stdout, process.stderr)

        logger.info("  Rendering complete!")

    finally:
        # 6. Clean up temporary ASS file
        if os.path.exists(ass_path):
            os.remove(ass_path)
            logger.info(f"  Cleaned up temporary file: {ass_path}")

    logger.info("=" * 80)
    logger.info("Video Synthesis Complete!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    return output_path


if __name__ == "__main__":
    import os

    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    material_json = os.path.join(current_dir, "output", "chapter1_material.json")
    output_video = os.path.join(current_dir, "output", "chapter1_video.mp4")

    try:
        synthesize_video_from_material(
            material_json_path=material_json,
            output_path=output_video,
            preview_duration=30,
        )
    except Exception as e:
        logger.error(f"Failed to synthesize video: {e}")
        raise
