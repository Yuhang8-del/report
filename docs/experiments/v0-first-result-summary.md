# v0 首版可运行结果摘要

## 1. 结果定位

本轮是面向科研实验与演示原型的首版完整链路：

`Teacher（全量人工标注） → 伪标签生成/过滤/审计 → Student（20%人工标签 + 伪标签） → 固定测试集 → PySide6 图像演示`。

本轮允许在正式伪标签精度门禁未通过时继续运行，目的是先形成一版可复现、可演示的端到端结果。该结果不宣称达到 mAP@0.5 0.80 或正式监督上界 0.85。

## 2. 固定协议

- 类别：Apple、Banana、Orange、Strawberry、Pineapple。
- 模型：YOLOv8m，输入尺寸 1024，RTX 3080，Conda 环境 `fruit-ssod`。
- 固定划分指纹：`7653d1f762053b90362803c8b2d25d287769de055fe11595565319f7fabe159c`。
- 固定测试集：90 张图像；测试结果由 `evaluations/test.json` 记录。
- 相机：未启用；开放世界识别：未启用，保留后续接口。

## 3. 端到端实验证据

### Teacher

运行目录：`E:/fruit_ssod_runtime/artifacts_v15/runs/supervised-v2-full-yolov8m-1024-teacher-seed42`

| 指标 | 验证集 | 固定测试集 |
|---|---:|---:|
| mAP@0.5 | 0.422360 | 0.317852 |
| mAP@0.5:0.95 | 0.255375 | 0.177501 |
| Precision | 0.478906 | 0.372837 |
| Recall | 0.433742 | 0.377064 |

Teacher 最佳权重：`weights/best.pt`，SHA-256：`2913b82aa9ef3206cb4f6092b58026ce5d6bec881c2de61aa1c17072e6a094c5`。

### 伪标签

- 双视图候选框：5,546 个。
- v0 信任过滤后保留：146 个。
- 审计 Precision：0.608696。
- 正式精度门禁结果：`refresh_allowed=false`；本轮因客户授权的 v0 探索开关继续执行，并在 Student provenance 中记录了该覆盖决策。

### Student

运行目录：`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-supervised-v2-teacher-seed42`

| 指标 | 验证集 | 固定测试集 |
|---|---:|---:|
| mAP@0.5 | 0.366312 | 0.263117 |
| mAP@0.5:0.95 | 0.220509 | 0.154698 |
| Precision | 0.512921 | 0.300340 |
| Recall | 0.379380 | 0.318192 |

Student 最佳权重：`weights/best.pt`，SHA-256：`7b87c8695cf4537899c4696f9b008f0862483452182e2d9e2f53827e9cc10e94`。

## 4. 关于“之前 0.6、现在 0.3”的说明

两者不是同一协议下的数值：

| 数值 | 来源 | 数据划分 | 含义 |
|---:|---|---|---|
| 0.570527 | v10 Teacher `run_record.json` | v10 独立划分 | 旧实验验证指标 |
| 0.607870 | v12 探索实验 `run_record.json` | v12 独立划分 | 旧实验验证指标 |
| 0.317852 | v2 Teacher `evaluations/test.json` | 当前 v2 固定测试集 | 当前 Teacher 的可比测试结果 |
| 0.263117 | v0 Student `evaluations/test.json` | 当前 v2 固定测试集 | 当前半监督 Student 的可比测试结果 |

因此，0.3 不是把同一模型的 0.6“突然变成了 0.3”，而是当前 Student 在新的固定测试协议下得到的结果。旧的 0.6 仍保留为历史探索记录，但不能用于证明当前模型达标。后续优化将始终同时报告验证集和固定测试集，避免再次混淆。

## 5. GUI 首版证据

已用当前 Student 权重完成 PySide6 离线图像推理导出：

`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r2`

该目录包含 3 张标注图、`detections.csv`、`results.json`、`manifest.json` 和 `v0_metadata.json`。相机和开放世界模式均为关闭状态。

## 6. 可复现入口

固定测试评估：

```powershell
$env:PYTHONPATH='E:/bishe/fruit/.worktrees/fruit-ssod-implementation/src'
E:/anaconda/envs/fruit-ssod/python.exe -m fruit_ssod.cli.evaluate_student_test `
  --run-dir E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-supervised-v2-teacher-seed42 `
  --data E:/fruit_ssod_runtime/data/fruit_ssod/processed/yolo/supervised_v2_100_seed42/dataset.yaml `
  --split-manifest E:/fruit_ssod_runtime/data/fruit_ssod/manifests/splits_v2/split_manifest.json `
  --device cuda:0
```

测试与质量检查：`410 passed, 1 skipped`。

## 7. 下一步

当前 v0 结果作为可运行基线保留；下一轮从数据质量、类别不平衡、增强策略和伪标签质量四个方向并行优化，仍使用同一固定测试协议，优化结果再与本摘要中的 v0 基线比较。

## 8. 第一轮持续优化对照

运行目录：`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-longer-supervised-v2-teacher-seed42-r1`

该轮保持相同的 v2 划分、20%人工标签和146个已审计伪标签，仅将训练上限提高到100轮、耐心值设为15并取消前10层冻结。训练在第17轮触发早停。

| 指标 | v0 Student | 优化轮 Student |
|---|---:|---:|
| 验证集 mAP@0.5 | 0.366312 | 0.342439 |
| 固定测试集 mAP@0.5 | 0.263117 | 0.266709 |
| 固定测试集 mAP@0.5:0.95 | 0.154698 | 0.149349 |
| 固定测试集 Precision | 0.300340 | 0.298437 |
| 固定测试集 Recall | 0.318192 | 0.367040 |

优化轮最佳权重 SHA-256：`f94a78f0d5f30ef1acca9202b58d771e231a35a4bc9b6660e4e19fc49c61e3d1`。该轮固定测试只有小幅 mAP@0.5 提升，说明单纯延长训练不是主要瓶颈；后续优先调整伪标签质量和数据覆盖率。

## 9. 严格伪标签优化对照

运行目录：`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-strict-pseudo-v2-teacher-seed42-r1`

本轮把验证校准目标精度从0.70提高到0.90。全训练池保留52个伪标签；受保护审计子集保留10个，审计 Precision 为0.900000，满足本轮探索刷新条件。训练在第12轮早停。

| 指标 | v0 Student | 严格伪标签轮 |
|---|---:|---:|
| 验证集 mAP@0.5 | 0.366312 | 0.326681 |
| 固定测试集 mAP@0.5 | 0.263117 | 0.246821 |
| 固定测试集 mAP@0.5:0.95 | 0.154698 | 0.139008 |
| 固定测试集 Precision | 0.300340 | 0.344778 |
| 固定测试集 Recall | 0.318192 | 0.297727 |

严格轮最佳权重 SHA-256：`9cb4ae6c7c40145505d1387168d3caba32e4db0612509fa37c44de5a7a2d3a11`。结果表明仅提高伪标签阈值会造成覆盖不足，当前不选它替代 v0 演示模型；后续重点改为扩大独立无标注池、增加有效标注覆盖并处理类别不平衡。

## 10. 数据覆盖优化（进行中）

已通过独立性校验封存2,205张 Open Images image-only 图像，并与当前 v2 Teacher 数据集完成 SHA-256 去重。与原136张无标注图合并后，新池共有2,341张图像；Teacher 产生25,036个双视图候选框，Trust Filter保留3,499个。当前 Student 运行目录为：

`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-independent-pool-v2-teacher-seed42-r1`

该轮尚未发布固定测试指标；完成后仍使用同一90张测试图、同一 split fingerprint 进行评估。

## 11. Independent-pool Student result (completed 2026-08-05)

The expanded independent-pool Student run completed 60 epochs and was
independently evaluated on the sealed 90-image fixed test. The measured result
is exploratory and is not presented as a pass of the customer target:

| Metric | Validation | Fixed test |
|---|---:|---:|
| mAP@0.5 | 0.38145 | 0.2831098263 |
| mAP@0.5:0.95 | 0.23257 | 0.1723647127 |
| Precision | 0.53959 | 0.4432110111 |
| Recall | 0.31602 | 0.2585118857 |

Run directory:
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-independent-pool-v2-teacher-seed42-r1`

Fixed-test evidence uses split fingerprint
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9`,
90 test images, and checkpoint SHA-256
`4ef41e4c81e1fbdb2755d7acd425fd25fb1c58309dae7c2d78cc9540469beea7`.
The no-class-threshold paired audit was prepared with post-filter Precision
`0.65625` (`refresh_allowed=false`). The first chained `r1` attempt terminated
after epoch 1 without a terminal state and was marked failed while preserving
its artifacts. A separate `r2` run with explicit stdout/stderr capture was
then completed and evaluated under the same fixed-test protocol.

## 12. Combined Student retry r2 (completed)

The retry run
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r2`
completed with early stopping after 31 recorded epochs. Its best validation
mAP@0.5 was `0.41341` at epoch 14. The fixed-test watcher generated a sealed
evaluation on the same 90-image test membership and split fingerprint:

| Metric | Validation best | Fixed test |
|---|---:|---:|
| mAP@0.5 | 0.41341 | 0.2741157961 |
| mAP@0.5:0.95 | not used for selection | 0.1618724549 |
| Precision | not used for selection | 0.3384724546 |
| Recall | not used for selection | 0.3613719988 |
| F1 | not used for selection | 0.3495475798 |

Checkpoint SHA-256:
`12fbaac37ffa5674a6f4447df18fc62092fea19d1f0cbc3e3610684cf10e30a3`.
This is exploratory evidence and remains below the customer target; it is
recorded as a separate run and does not overwrite the earlier r1 result.

## 13. Natural-unresampled Student comparison (completed)

The natural-unresampled comparison kept the same Teacher checkpoint, combined
training pool, pseudo-label snapshot and fixed-test protocol while removing the
50/50 exposure resampling. It stopped after 42 recorded epochs; the best
validation checkpoint was epoch 32 with validation mAP@0.5 `0.44386`.

| Metric | Fixed test |
|---|---:|
| mAP@0.5 | 0.2872889802 |
| mAP@0.5:0.95 | 0.1759981120 |
| Precision | 0.3459670219 |
| Recall | 0.3586910823 |
| F1 | 0.3522141724 |

Run directory:
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-natural-unresampled-combined-v2-teacher-seed42-r1`

Checkpoint SHA-256:
`40873efeaf5dd2da18a2116bd6147e9b957c1f795bb4b3efeb00404e98c36f30`.
This is the strongest fixed-test Student result so far, but it is still an
exploratory result rather than a claim that the customer target is met.

The independent GUI export is available at
`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r3`.
It records the same run ID and checkpoint hash, contains three annotated image
results and keeps camera and open-world actions disabled.

The first detached launch of the low-learning-rate continuation stopped after
its first validation without a terminal run state; its partial artifacts are
preserved and excluded from comparison. A fresh run with the same configuration
and a new run ID
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2` is active with
explicit logs and a fixed-test watcher.

The first-result policy no longer blocks on a target threshold. A serial queue
is active so that, after r2 is evaluated, seed-43 will start automatically
under a new run ID with separate logs; the r3 GUI candidate remains preserved
as the current runnable delivery version.

## 8. 首版结果已固化，优化继续

`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2` 已完成训练并经
固定测试验证。训练因早停在 43 个记录轮次后结束，验证集最佳 mAP@0.5 为
`0.45096`（第 28 轮）。固定测试使用 90 张测试图像，mAP@0.5 为
`0.2985708107`，mAP@0.5:0.95 为 `0.1723392126`，Precision 为
`0.3222840638`，Recall 为 `0.3796603397`，F1 为 `0.3486272603`。

首版模型及其证据位于：

- checkpoint：
  `E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2/weights/best.pt`
- checkpoint SHA-256：
  `f96cf93c4deeff9f93a89c467038ae3a9d79700b80b3196194efcd4098acbcb8`
- fixed-test JSON：该 run 目录下的 `evaluations/test.json`
- PySide6 离线演示导出：
  `E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r4`

该结果作为客户授权的“先出一版最终结果”交付基线保存；由于指标未达到
目标值，报告中如实标注为探索性结果，但不暂停后续优化。seed43 已按相同
数据、伪标签快照和测试协议串行启动，后续结果将以独立 run ID 追加记录。
