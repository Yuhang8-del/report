# Delivery Manifest Contract

The release package is built outside Git after final QA. Its generated
`delivery-manifest.json` must enumerate every included regular file with:

- relative path;
- byte count;
- SHA-256 digest.

The package must include, where final evidence exists:

1. Dataset audit and data manifests (not redistributable source archives).
2. Effective experiment configurations, run records, result package and
   acceptance evidence.
3. Released model checkpoint with its SHA-256 and the matching RTX 3080
   benchmark record.
4. Pseudo-label and Trust Filter audit records.
5. PySide6 application source, Windows launchers, user guide and QA checklist.
6. Final report source, immutable `report_data.json`, report assets, DOCX and
   PDF.
7. Reproduction instructions and the exact Git revision.

The package must exclude credentials, `.env` files, private keys, temporary
directories, unlicensed source redistribution and unrelated training caches.
`scripts/build_delivery_manifest.ps1` fails when it encounters common
credential filenames or a symbolic link. The release reviewer must still
confirm licensing and content suitability before distribution.
