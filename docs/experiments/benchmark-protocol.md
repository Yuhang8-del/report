# RTX 3080 inference benchmark protocol

This benchmark is a deployment measurement for the final five-class fruit detector. It is separate from mAP evaluation and must not be used to alter the fixed test protocol.

Run it only after selecting the final, completed checkpoint and closing other GPU-heavy applications. The default protocol uses a square synthetic RGB image at **640 × 640**, 20 warm-up inferences, then 100 measured inferences on `cuda:0`. CUDA synchronization is mandatory immediately before and after every measured call, so reported latency is not distorted by asynchronous kernel launch.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.benchmark_model `
  --weights <final-best.pt> `
  --output <artifact-root>\exports\final_benchmark.json `
  --device cuda:0
```

Use `--dry-run` first to validate the checkpoint and output path without importing CUDA or running a model. It does not create the output JSON. Existing output files are never overwritten; publish a new path for a new benchmark.

The emitted JSON seals the following evidence:

- warm-up and measured iteration counts, synchronization policy, CUDA device, and image size;
- mean, median, p95, minimum and maximum latency (milliseconds), plus FPS calculated from mean latency;
- peak PyTorch allocated CUDA memory (bytes and MiB);
- the caller-supplied checkpoint path plus the SHA-256, bytes and MiB of the
  private checkpoint snapshot actually loaded by Ultralytics. The source is
  hashed before and after copying, and a temporary B-to-A swap is rejected;
- GPU name, NVIDIA driver version, PyTorch version, CUDA runtime and Ultralytics version.
- the explicit CUDA logical GPU index and UUID, cross-checked against the full `nvidia-smi` physical GPU UUID table; logical-to-physical ordinal remapping is rejected. The record also contains active compute PIDs, the benchmark PID, and the enforced `no_foreign_compute_processes` policy. GPU identity, driver/runtime fields, and selected-GPU process isolation are sampled both before and after timing; any drift or new foreign compute PID invalidates the run.

The benchmark refuses CPU execution, implicit CUDA-device selection, unavailable CUDA, a GPU other than the RTX 3080, a PyTorch/`nvidia-smi` GPU-name mismatch, missing driver evidence, a foreign compute process on the selected GPU, malformed/empty weights, a non-five-class checkpoint, and attempts to overwrite a result. Record any such failure in the experiment ledger rather than substituting an unverified throughput number.
