# Local chapter 0 media

These files mirror the media embedded in the chapter 0 notebooks. Original source URLs and SHA-256 hashes are recorded in `manifest.json`; attribution and rights remain with the original sources. Five outdated `*-v2.html` links were corrected to the existing upstream filenames before mirroring.

HTML resources use local JavaScript. Lecture videos hosted by YouTube remain optional external links; the small course animations are included locally. Dataset/model downloads are separate and live in the ignored `exercises/data/` cache.

Maintain these assets with `scripts/chapter0_assets.py` from the repository root. `--check` works offline. The master files retain upstream URLs for provenance; generation rewrites them to relative paths for notebooks.
