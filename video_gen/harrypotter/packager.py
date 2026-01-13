"""Packager for Harry Potter audiobook output files."""

import json
import os
import zipfile
from datetime import datetime

from loguru import logger

from video_gen.harrypotter.models import AudiobookChapterMaterial


class OutputPackager:
    """Package output files (videos, images, materials, caches) into a zip archive.

    This class handles packaging all generated files from the Harry Potter audiobook
    processing pipeline into a single zip file for easy distribution and backup.

    Example:
        >>> packager = OutputPackager()
        >>> zip_path = packager.package(
        ...     output_dir="output",
        ...     results={1: material1, 2: material2},
        ...     cache_dir="/tmp/cached",
        ...     include_cache=True
        ... )
        >>> print(f"Package created: {zip_path}")
    """

    def package(
        self,
        output_dir: str,
        results: dict[int, AudiobookChapterMaterial],
        cache_dir: str = "/tmp/cached",
        include_cache: bool = True,
    ) -> str:
        """Package all output files into a zip archive.

        Uses the output_files and cache_files tracked in each AudiobookChapterMaterial
        to package only the files that were actually generated, avoiding any stale
        or unrelated files in the output directory.

        Args:
            output_dir: The output directory (used for zip file location)
            results: Dictionary mapping chapter_id to AudiobookChapterMaterial
            cache_dir: Path to cache directory. Default is /tmp/cached (legacy, not used)
            include_cache: Whether to include cache files. Default is True

        Returns:
            str: Path to the created zip file

        Raises:
            OSError: If there are issues creating the zip file or accessing files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chapter_ids_str = "_".join(str(cid) for cid in sorted(results.keys()))
        zip_filename = f"harrypotter_chapters_{chapter_ids_str}_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        # Collect all output files and cache files from materials
        all_output_files = []
        all_cache_files = []
        for material in results.values():
            all_output_files.extend(material.output_files)
            if include_cache:
                all_cache_files.extend(material.cache_files)

        logger.info("=" * 80)
        logger.info("Packaging Output Files (Tracked Files Only)")
        logger.info("=" * 80)
        logger.info(f"  Chapters: {len(results)}")
        logger.info(f"  Output files: {len(all_output_files)}")
        logger.info(f"  Cache files: {len(all_cache_files) if include_cache else 0}")
        logger.info(f"  Zip file: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add output files from tracked paths
            self._add_tracked_files(zipf, all_output_files, output_dir, "output")

            # Add cache files if requested
            if include_cache:
                self._add_tracked_files(zipf, all_cache_files, cache_dir, "cache")

            # Add manifest
            self._add_manifest(zipf, timestamp, results, include_cache, all_output_files, all_cache_files)

        file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info("=" * 80)
        logger.info("✅ Package Created Successfully!")
        logger.info(f"  Path: {zip_path}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")
        logger.info("=" * 80)

        return zip_path

    def _add_tracked_files(
        self, zipf: zipfile.ZipFile, file_paths: list[str], base_dir: str, archive_prefix: str
    ) -> None:
        """Add tracked files to the zip archive.

        Args:
            zipf: The ZipFile object to add files to
            file_paths: List of absolute file paths to add
            base_dir: Base directory for calculating relative paths
            archive_prefix: Prefix for archive paths (e.g., "output" or "cache")
        """
        logger.info(f"Adding {len(file_paths)} {archive_prefix} files...")
        added_count = 0
        skipped_count = 0

        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"  File not found (skipping): {file_path}")
                skipped_count += 1
                continue

            # Calculate relative path for archive
            try:
                # Try to make it relative to base_dir
                arcname = os.path.relpath(file_path, base_dir)
            except ValueError:
                # If file is on a different drive or not under base_dir,
                # use just the filename
                arcname = os.path.basename(file_path)

            # Add file to archive
            zipf.write(file_path, arcname=f"{archive_prefix}/{arcname}")
            logger.debug(f"  Added: {archive_prefix}/{arcname}")
            added_count += 1

        logger.info(f"  Successfully added: {added_count} files")
        if skipped_count > 0:
            logger.warning(f"  Skipped (not found): {skipped_count} files")

    # Legacy method - no longer used, kept for backward compatibility
    def _add_cache_files_legacy(self, zipf: zipfile.ZipFile, cache_dir: str) -> None:
        """Add cache files to the zip archive (legacy method).

        This method is no longer used. Cache files are now tracked in
        AudiobookChapterMaterial.cache_files and added via _add_tracked_files.

        Args:
            zipf: The ZipFile object to add files to
            cache_dir: Path to cache directory
        """
        logger.warning("Legacy _add_cache_files method called - this should not happen")
        logger.info("Cache files are now tracked in material.cache_files")

    def _add_manifest(
        self,
        zipf: zipfile.ZipFile,
        timestamp: str,
        results: dict[int, AudiobookChapterMaterial],
        include_cache: bool,
        all_output_files: list[str],
        all_cache_files: list[str],
    ) -> None:
        """Add a manifest file with metadata to the zip archive.

        Args:
            zipf: The ZipFile object to add files to
            timestamp: Timestamp string for the manifest
            results: Dictionary mapping chapter_id to AudiobookChapterMaterial
            include_cache: Whether cache files were included
            all_output_files: List of all output file paths
            all_cache_files: List of all cache file paths
        """
        manifest_content = {
            "created_at": timestamp,
            "chapters": list(results.keys()),
            "chapter_details": {
                str(cid): {
                    "title": material.chapter.title,
                    "duration": material.chapter.duration,
                    "transcript_segments": len(material.transcript),
                    "scenes": len(material.scenes),
                    "output_files_count": len(material.output_files),
                    "cache_files_count": len(material.cache_files),
                }
                for cid, material in results.items()
            },
            "include_cache": include_cache,
            "total_output_files": len(all_output_files),
            "total_cache_files": len(all_cache_files) if include_cache else 0,
        }

        manifest_json = json.dumps(manifest_content, indent=2, ensure_ascii=False)
        zipf.writestr("manifest.json", manifest_json)
        logger.info("Added manifest.json")
