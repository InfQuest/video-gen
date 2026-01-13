"""Scene detection and image prompt generation for Harry Potter audiobook.

This module analyzes transcript segments to identify scene changes and generates
first-person POV image generation prompts for each unique scene.
"""

import json
import os
import re
from typing import List

import requests
from loguru import logger
from pydantic import BaseModel

from video_gen.core.configs.config import settings
from video_gen.core.tools.openai_client import OpenAIClient
from video_gen.core.tools.volcengine_client import VolcengineImageClient
from video_gen.common.tools import cached
from video_gen.video_material import TranscriptSegment


class Scene(BaseModel):
    """Scene definition (reusable)"""

    idx: int
    description: str
    image_prompt: str
    scene_type: str | None = None
    image_url: str | None = None  # Generated image URL (local path or S3 URL)


class SceneSegment(BaseModel):
    """Segment to scene mapping"""

    start_segment: int
    end_segment: int
    start_time: float
    end_time: float
    scene_idx: int


class SceneDetectionResult(BaseModel):
    """Complete scene detection result"""

    scenes: List[Scene]
    scene_segments: List[SceneSegment]


# LLM Prompt for scene detection
SCENE_DETECTION_PROMPT = """You are an expert in analyzing Harry Potter audiobook transcripts to identify scene changes.

Input: A numbered list of transcript segments from Harry Potter audiobook

Task: Identify distinct scenes and generate image generation prompts. Output TWO parts:
1. Scenes list (unique scene definitions that can be reused)
2. Scene-segment mappings (which segments use which scene)

Output Requirements:
- Return JSON object with two keys: "scenes" and "scene_segments"
- "scenes": Array of unique scene definitions
  - Each scene has: idx (integer), description (string), image_prompt (string)
  - Same scene (e.g., "Privet Drive street") should appear only once
- "scene_segments": Array of segment-to-scene mappings
  - Each mapping has: start_segment (integer), end_segment (integer), scene_idx (integer)
  - scene_idx references the idx in "scenes" array

Scene Detection Rules:
1. Look for location changes (e.g., "Privet Drive" → "Office")
2. Look for time changes (e.g., morning → evening)
3. Look for significant action shifts
4. Create as many scenes as needed to capture all significant changes - no upper limit
5. Segments can be 2-8 segments long (flexibility for natural scene boundaries)
6. Prefer more granular scenes over fewer long scenes for visual variety
7. **Reuse scenes**: If the narrative returns to the same location, reference the existing scene_idx

Image Prompt Requirements:
1. Based ONLY on book descriptions - DO NOT invent details
2. **Harry Potter Scene prefix**: MUST start with "Harry Potter scene:"
3. **Detailed scene description**: Describe the environment, lighting, atmosphere, key objects, and spatial layout in detail
4. **Style**: "Hand-drawn watercolor illustration, architectural sketch style, pen and watercolor technique, soft color rendering, dramatic sky, Studio Ghibli inspired atmosphere, concept art quality"
5. **First-person POV**: What the viewer would see from their own eyes, immersive perspective
6. **CRITICAL - EMPTY SCENE ONLY**:
   - ABSOLUTELY NO main subjects, protagonists, or focal characters
   - ABSOLUTELY NO animals of any kind (no cats, owls, birds, pets)
   - Background pedestrians ONLY if absolutely necessary for street scenes, and they must be:
     * Tiny, distant, out of focus
     * Walking away or facing away from camera
     * Rendered as simple silhouettes or blurred shapes
     * NEVER the focus or in the foreground
   - Focus on ARCHITECTURE, ENVIRONMENT, and ATMOSPHERE only
   - **Text/Signage**: Any visible text, signs, or writing MUST be in English only (Harry Potter takes place in Britain). NO Chinese, Japanese, or other non-English text
   - End every prompt with: "Empty environment scene, no main characters, no animals, no people in focus, pure location background. Any visible text in English only."
7. Focus on setting/atmosphere to create presence and immersion
8. Use canonical Harry Potter visual elements (architecture, objects, atmosphere)
9. Keep prompts detailed (70-100 words), always include "Harry Potter scene:" prefix, detailed scene description, style keywords, and the "Empty environment" ending

Example Output:
{
  "scenes": [
    {
      "idx": 0,
      "description": "Privet Drive suburban street, early morning",
      "image_prompt": "Harry Potter scene: First-person view of a perfectly ordinary suburban British street at dawn. Looking down a neat row of identical houses with tidy front gardens, trimmed hedges, and pristine driveways. The house with 'number four' on the door is prominently visible with its pale walls and polished brass fixtures. Dramatic cloudy grey sky overhead creates an oppressive atmosphere. Focus on architecture and street layout. Hand-drawn watercolor illustration, architectural sketch style, pen and watercolor technique, soft color rendering, dramatic sky, Studio Ghibli inspired atmosphere, concept art quality. Empty environment scene, no main characters, no animals, no people in focus, pure location background. Any visible text in English only."
    },
    {
      "idx": 1,
      "description": "Corporate office interior",
      "image_prompt": "Harry Potter scene: First-person POV inside a sterile corporate office with grey walls and fluorescent lighting. Looking at a large wooden desk with a black telephone, neat stack of papers, and leather executive chair. Large window behind the desk shows a city skyline with other office buildings. Morning sunlight streams through the window creating sharp shadows. Empty chair, quiet interior space. Hand-drawn watercolor illustration, architectural sketch style, pen and watercolor technique, soft color rendering, dramatic lighting, Studio Ghibli inspired atmosphere, concept art quality. Empty environment scene, no main characters, no animals, no people in focus, pure location background. Any visible text in English only."
    },
    {
      "idx": 2,
      "description": "Car interior, commuting",
      "image_prompt": "Harry Potter scene: First-person view from the driver's seat of a car interior. Looking through the windshield at suburban British streets with houses and parked cars. Dashboard visible at the bottom with speedometer and steering wheel partially in frame. Empty passenger seat to the right with seat belt hanging. Rearview mirror shows the road behind. Morning commute atmosphere with grey overcast sky. Hand-drawn watercolor illustration, architectural sketch style, pen and watercolor technique, soft color rendering, atmospheric perspective, Studio Ghibli inspired atmosphere, concept art quality. Empty environment scene, no main characters, no animals, no people in focus, pure location background. Any visible text in English only."
    }
  ],
  "scene_segments": [
    {
      "start_segment": 1,
      "end_segment": 5,
      "scene_idx": 0
    },
    {
      "start_segment": 6,
      "end_segment": 10,
      "scene_idx": 1
    },
    {
      "start_segment": 11,
      "end_segment": 15,
      "scene_idx": 2
    },
    {
      "start_segment": 16,
      "end_segment": 20,
      "scene_idx": 0
    }
  ]
}

Important:
- Output ONLY valid JSON object
- No markdown, no code blocks, no explanations
- description and image_prompt in English
- Ensure start_segment < end_segment
- No gaps or overlaps between scene_segments
- scene_idx must reference a valid scene in the "scenes" array
"""  # noqa: E501


@cached(cache_dir="/tmp/cached", exclude_params=["llm_client"])
def detect_scenes_with_prompts(
    transcript: List[TranscriptSegment],
    llm_client: "OpenAIClient",
) -> SceneDetectionResult:
    """Detect scenes and generate image prompts using LLM.

    Args:
        transcript: List of transcript segments
        llm_client: OpenAI-compatible client (configured with model)

    Returns:
        SceneDetectionResult with scenes and scene_segments

    Raises:
        ValueError: If LLM returns invalid JSON or None response
    """
    logger.info("Starting scene detection with LLM...")
    logger.info(f"Analyzing {len(transcript)} transcript segments")

    # Format transcript for LLM
    transcript_text = "\n".join([f"[{i + 1}] {seg.text}" for i, seg in enumerate(transcript)])

    user_input = f"""Transcript Segments:

{transcript_text}

Analyze the above transcript and identify distinct scenes. Return JSON with "scenes" and "scene_segments"."""

    # Call LLM
    logger.info("Calling LLM for scene detection...")
    response = llm_client.generate(
        instruction=SCENE_DETECTION_PROMPT,
        user_input=user_input,
    )

    if response is None:
        raise ValueError("LLM returned None response")

    logger.info(f"Received LLM response: {len(response)} characters")

    # Parse JSON response
    try:
        # Clean up response (remove markdown code blocks if present)
        response_clean = response.strip()
        if response_clean.startswith("```"):
            # Remove markdown code blocks
            lines = response_clean.split("\n")
            response_clean = "\n".join(line for line in lines if not line.startswith("```"))

        data = json.loads(response_clean)
        logger.info("Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response: {response[:500]}...")
        raise ValueError(f"LLM returned invalid JSON: {e}")

    # Validate structure
    if "scenes" not in data or "scene_segments" not in data:
        raise ValueError(f"LLM response missing required keys. Got: {list(data.keys())}")

    # Parse scenes
    scenes = [Scene(**scene_data) for scene_data in data["scenes"]]
    logger.info(f"Parsed {len(scenes)} unique scenes")

    # Parse scene segments and add timing information
    scene_segments = []
    for seg_data in data["scene_segments"]:
        # Validate required fields
        if "start_segment" not in seg_data or "end_segment" not in seg_data or "scene_idx" not in seg_data:
            logger.warning(f"Skipping scene segment with missing required fields: {seg_data}")
            continue

        start_idx = seg_data["start_segment"] - 1  # Convert to 0-indexed
        end_idx = seg_data["end_segment"] - 1

        # Validate indices
        if start_idx < 0 or end_idx >= len(transcript):
            logger.warning(f"Invalid segment range: {start_idx + 1}-{end_idx + 1}, skipping")
            continue

        # Get timing from transcript
        start_time = transcript[start_idx].start_at
        end_time = transcript[end_idx].end_at

        scene_segment = SceneSegment(
            start_segment=seg_data["start_segment"],
            end_segment=seg_data["end_segment"],
            start_time=start_time,
            end_time=end_time,
            scene_idx=seg_data["scene_idx"],
        )
        scene_segments.append(scene_segment)

    logger.info(f"Parsed {len(scene_segments)} scene-segment mappings")

    # Validate scene indices
    scene_indices = {scene.idx for scene in scenes}
    for seg in scene_segments:
        if seg.scene_idx not in scene_indices:
            logger.warning(f"Scene segment references non-existent scene_idx: {seg.scene_idx}")

    result = SceneDetectionResult(scenes=scenes, scene_segments=scene_segments)

    # Log scene reuse statistics
    scene_usage = {}
    for seg in scene_segments:
        scene_usage[seg.scene_idx] = scene_usage.get(seg.scene_idx, 0) + 1

    logger.info("Scene reuse statistics:")
    for scene in scenes:
        usage = scene_usage.get(scene.idx, 0)
        logger.info(f"  Scene {scene.idx}: '{scene.description}' used {usage} times")

    return result


class SceneDetector:
    """Scene detector for Harry Potter audiobook transcripts."""

    def __init__(self, llm_client=None):
        """Initialize scene detector.

        Args:
            llm_client: Optional OpenAI-compatible client. If None, creates default client.
        """
        if llm_client is None:
            self.llm_client = OpenAIClient(
                api_key=settings.openai_api_key,
                model_name="google/gemini-2.5-pro",
            )
        else:
            self.llm_client = llm_client

    def detect_scenes(self, transcript: List[TranscriptSegment]) -> SceneDetectionResult:
        """Detect scenes in transcript.

        Args:
            transcript: List of transcript segments

        Returns:
            SceneDetectionResult with scenes and scene_segments
        """
        return detect_scenes_with_prompts(transcript=transcript, llm_client=self.llm_client)

    def generate_scene_images(
        self,
        result: SceneDetectionResult,
        output_dir: str = "output/scene_images",
        model: str = "doubao-seedream-4-0-250828",
        size: str = "2K",
        limit: int | None = None,
    ) -> SceneDetectionResult:
        """Generate images for all scenes using Volcengine ARK API.

        Args:
            result: SceneDetectionResult with scenes
            output_dir: Directory to save generated images
            model: Image generation model to use (e.g., "doubao-seedream-4-0-250828")
            size: Image size (1K, 2K, or 4K)
            limit: Maximum number of scenes to generate images for (None = all)

        Returns:
            Updated SceneDetectionResult with image_url populated

        Note:
            Uses Volcengine ARK image generation API.
            Images are saved locally and paths are stored in scene.image_url
        """
        os.makedirs(output_dir, exist_ok=True)

        # Limit scenes if specified
        scenes_to_process = result.scenes[:limit] if limit else result.scenes

        logger.info(f"Generating images for {len(scenes_to_process)} scenes...")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {model}")
        logger.info(f"Size: {size}")
        if limit:
            logger.info(f"Limit: {limit} scenes (out of {len(result.scenes)} total)")

        # Create image generation client
        image_client = VolcengineImageClient()

        updated_scenes = []
        for scene in scenes_to_process:
            try:
                output_path = os.path.join(output_dir, f"scene_{scene.idx:03d}.png")

                # Check if image already exists locally
                if os.path.exists(output_path):
                    logger.info(f"Image already exists for Scene {scene.idx}: {scene.description}")
                    logger.info(f"  📁 Using existing: {output_path}")
                    image_url = output_path
                else:
                    logger.info(f"Generating image for Scene {scene.idx}: {scene.description}")

                    # Call image generation API
                    image_url = self._generate_single_image_volcengine(
                        client=image_client,
                        prompt=scene.image_prompt,
                        output_path=output_path,
                        model=model,
                        size=size,
                    )

                    logger.info(f"  ✅ Saved to: {image_url}")

                # Update scene with image URL
                updated_scene = scene.model_copy(update={"image_url": image_url})
                updated_scenes.append(updated_scene)

            except Exception as e:
                logger.error(f"  ❌ Failed to generate image for Scene {scene.idx}: {e}")
                updated_scenes.append(scene)  # Keep original without image_url

        # Return updated result
        return SceneDetectionResult(scenes=updated_scenes, scene_segments=result.scene_segments)

    def _generate_single_image(self, client: OpenAIClient, prompt: str, output_path: str) -> str:
        """Generate a single image using chat completions API.

        Args:
            client: OpenAI client configured for image generation
            prompt: Image generation prompt
            output_path: Local path to save the image

        Returns:
            Local file path to the saved image

        Raises:
            Exception: If image generation or download fails

        Note:
            Uses OpenAIClient.generate() with chat completions API for image generation.
            The model (google/gemini-2.5-flash-image) returns image URLs in the response.
        """
        # Call image generation using existing client
        instruction = (
            "You are an image generation assistant. "
            "Generate an image based on the user's prompt and return only the image URL."
        )
        response_content = client.generate(
            instruction=instruction,
            user_input=prompt,
        )

        if not response_content:
            raise ValueError("Empty response from image generation API")

        # Parse response to extract image URL
        # The response format may vary, try to extract URL
        image_url = None

        # Try to parse as JSON first (some models return JSON with image URL)
        try:
            if response_content.strip().startswith("{"):
                data = json.loads(response_content)
                if "url" in data:
                    image_url = data["url"]
                elif "image_url" in data:
                    image_url = data["image_url"]
        except json.JSONDecodeError:
            pass

        # If not JSON, check if it's a direct URL
        if not image_url:
            # Check if response is a URL
            if response_content.startswith("http://") or response_content.startswith("https://"):
                image_url = response_content.strip()

        # If still no URL, try to extract from markdown format
        if not image_url:
            # Check for markdown image format: ![alt](url)
            markdown_pattern = r"!\[.*?\]\((https?://[^\)]+)\)"
            match = re.search(markdown_pattern, response_content)
            if match:
                image_url = match.group(1)

        if not image_url:
            raise ValueError(f"Could not extract image URL from response: {response_content[:200]}...")

        # Download image from URL
        logger.info(f"Downloading image from: {image_url}")
        image_response = requests.get(image_url, timeout=60)
        image_response.raise_for_status()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(image_response.content)

        return output_path

    def _generate_single_image_volcengine(
        self, client: VolcengineImageClient, prompt: str, output_path: str, model: str, size: str
    ) -> str:
        """Generate a single image using Volcengine ARK API with retry logic.

        Args:
            client: Volcengine image client
            prompt: Image generation prompt
            output_path: Local path to save the image
            model: Model name to use
            size: Image size (1K, 2K, or 4K)

        Returns:
            Local file path to the saved image

        Raises:
            Exception: If image generation or download fails after all retries
        """
        import time

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Call image generation API
                image_url = client.generate_image(
                    prompt=prompt,
                    model=model,
                    size=size,
                    response_format="url",
                    watermark=True,
                )

                # Download image from URL
                logger.info(f"Downloading image from: {image_url}")
                image_response = requests.get(image_url, timeout=60)
                image_response.raise_for_status()

                # Save to file
                with open(output_path, "wb") as f:
                    f.write(image_response.content)

                return output_path

            except Exception as e:
                logger.warning(f"Image generation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} retry attempts failed for image generation")
                    raise  # Re-raise the exception after all retries


if __name__ == "__main__":
    import argparse

    from video_gen.harrypotter.models import AudiobookChapterMaterial

    parser = argparse.ArgumentParser(description="Test scene detection and image generation")
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate images for detected scenes",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=20,
        help="Number of segments to test (default: 20)",
    )
    args = parser.parse_args()

    # Load test material (use full chapter material)
    material_path = "src/materials/videos/harrypotter/output/chapter1_material.json"
    with open(material_path) as f:
        material = AudiobookChapterMaterial.model_validate_json(f.read())

    # Use limited segments for quick test
    test_transcript = material.transcript[: args.segments]

    # Detect scenes
    detector = SceneDetector()
    result = detector.detect_scenes(test_transcript)

    # Print results
    print("\n" + "=" * 80)
    print(f"Detected {len(result.scenes)} unique scenes:")
    print("=" * 80)
    for scene in result.scenes:
        print(f"\nScene {scene.idx}:")
        print(f"  Description: {scene.description}")
        print(f"  Prompt: {scene.image_prompt}")

    print("\n" + "=" * 80)
    print(f"Scene-Segment Mappings ({len(result.scene_segments)}):")
    print("=" * 80)
    for seg in result.scene_segments:
        scene = next(s for s in result.scenes if s.idx == seg.scene_idx)
        print(f"\nSegments {seg.start_segment}-{seg.end_segment} ({seg.start_time:.1f}s - {seg.end_time:.1f}s):")
        print(f"  Scene: {scene.description}")

    # Generate images if requested
    if args.generate_images:
        print("\n" + "=" * 80)
        print("Generating scene images...")
        print("=" * 80)

        result_with_images = detector.generate_scene_images(
            result=result, output_dir="src/materials/videos/harrypotter/output/scene_images"
        )

        print("\n" + "=" * 80)
        print("Image Generation Complete:")
        print("=" * 80)
        for scene in result_with_images.scenes:
            status = "✅" if scene.image_url else "❌"
            print(f"{status} Scene {scene.idx}: {scene.image_url or 'Failed'}")

        # Update result with images
        result = result_with_images

    # Save result
    output_path = "src/materials/videos/harrypotter/output/test_scene_detection.json"
    with open(output_path, "w") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"\n✅ Saved result to {output_path}")
