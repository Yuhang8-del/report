# Open Images acquisition

This project supports a deliberately small, explicit Open Images V7 acquisition
pipeline. It does **not** mean that Open Images data has already been downloaded.
The source is [Open Images](https://storage.googleapis.com/openimages/web/index.html),
whose dataset documentation and attribution requirements are described by the
[official Open Images site](https://github.com/openimages/dataset). Preserve the
source-image IDs, source URL manifest, and relevant source/license information
when preparing data for use.

The converter resolves only the exact `open_images_v7` display-name aliases in
the canonical registry. It filters `IsDepiction`, `IsInside`, and `IsGroupOf`
rows. For finite normalized coordinates outside the image, it clamps each
endpoint to `[0, 1]`; rows that then have zero or negative area are rejected.
It never guesses a class ID or relabels a source category.

## Local fixture run

The checked-in CSV fixtures and `tiny.svg` image are fully local. The fixture
URL map uses a relative `file:tiny.svg` reference, resolved relative to that
CSV, so this command converts labels without downloading anything:

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.download_open_images `
  --annotations tests/fixtures/open_images/annotations.csv `
  --class-descriptions tests/fixtures/open_images/class-descriptions.csv `
  --image-url-map tests/fixtures/open_images/image-urls.csv `
  --output-root .local/open-images-fixture --max-images 2
```

Use `--dry-run` to validate inputs without creating the output directory or
accessing URLs. `--report path/to/report.json` is the sole permitted dry-run
write destination.

Append `--download` to the local fixture command to exercise the downloader
against the checked-in `file:` image only; it makes no network request.

## Future shared-storage smoke run

Only after the shared storage and approved URL manifest are accessible, make a
five-image-per-class manifest (or one separately reviewed manifest per class)
and run the command below. The URL map is mandatory and configurable; no
storage URL is guessed or hard-coded.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.download_open_images `
  --annotations <approved-annotations.csv> `
  --class-descriptions <official-class-descriptions.csv> `
  --image-url-map <approved-five-per-class-url-map.csv> `
  --output-root <caller-chosen-output-root> --max-images 25 --download
```

Downloads are explicit (`--download`), resumable through `.part` files when a
server honours `Range`, atomically promoted after completion, and appended to
`download-ledger.jsonl` with SHA-256 checksums. Existing images are never
overwritten or deleted; a rerun records their checksum as `existing`.

## Source split provenance

`select_open_images` defaults to `--image-split train`. For a separately
versioned recovery dataset that intentionally samples the official validation
or test *source* pools before creating a new project-level split, set
`--image-split validation` or `--image-split test`. The selected manifest
records this source split and emits the matching official bucket URL. Never
mix a source-split label with an existing project-level held-out split.
