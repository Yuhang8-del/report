# Final QA Checklist

Use this checklist only after the final result package and release assets have
been generated. A checked item requires its corresponding command output or
manual observation to be retained in the release record.

## Automated checks

- [ ] Activate the documented Windows Conda environment and run
  `scripts/run_all_checks.ps1`.
- [ ] Re-run the dataset audit and verify zero critical findings.
- [ ] Verify all published result-package and report-asset SHA-256 values.
- [ ] Confirm all required primary seeds completed under one split fingerprint.
- [ ] Confirm the acceptance artifact is generated from the aggregate, not
  manually edited.
- [ ] Confirm the designated final checkpoint is the benchmarked checkpoint.
- [ ] Confirm no private credentials, source archives, datasets, weights or
  large runtime outputs are staged in Git.

## Demonstrator checks

- [ ] Launch `scripts/start_gui.ps1 -PreflightOnly` successfully.
- [ ] Test a valid single image, folder and video input.
- [ ] Test cancellation, export and invalid-input error messages.
- [ ] Confirm the GUI exposes no camera control.
- [ ] Confirm the open-world control is absent or disabled and documented as
  future work.

## Report and delivery checks

- [ ] Run final-report preflight against the immutable `report_data.json` and
  report-asset manifest.
- [ ] Render the generated DOCX to PDF and inspect all report pages at 100%.
- [ ] Confirm the main body is at most 5000 words and figures/tables are each
  at most ten.
- [ ] Confirm captions, cross-references, literature references and the Impact
  Statement are present.
- [ ] Confirm no camera, unsupported open-world completion, invented metric or
  unsupported conclusion is present.
- [ ] Generate the delivery manifest and verify every listed release hash.
