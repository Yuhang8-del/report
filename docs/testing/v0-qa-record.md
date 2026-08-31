# v0 实验与演示 QA 记录

本记录只覆盖客户要求的首版可运行原型，不代表正式多种子矩阵或最终报告已完成。

## 自动化结果

- `fruit-ssod` 环境：Python 3.10.20、PyTorch 2.5.1+cu121、CUDA RTX 3080。
- 完整 Python/GUI 测试：`410 passed, 1 skipped`。
- 无 GUI 自动化检查：`378 passed, 1 skipped`，由 `scripts/run_all_checks.ps1 -SkipGuiTests` 通过。
- 固定测试评估器验证了 Student 运行状态、checkpoint SHA-256、split fingerprint 和90张测试图成员关系。

## 演示结果

- PySide6 离线图像导出：`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r2`。
- 导出内容：3张标注图、`detections.csv`、`results.json`、`manifest.json`、`v0_metadata.json`。
- 相机：关闭；开放世界识别：关闭并预留接口。

## r2 演示候选更新

The GUI candidate now loads the completed Student retry checkpoint from
`ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r2`. Its checkpoint
SHA-256 is
`12fbaac37ffa5674a6f4447df18fc62092fea19d1f0cbc3e3610684cf10e30a3` and the
offline export is
`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r2`.
The export contains three annotated images, `detections.csv`, `results.json`,
`manifest.json` and `v0_metadata.json`; camera and open-world actions remain
disabled.

## r3 GUI candidate update

The natural-unresampled Student checkpoint was evaluated on the sealed 90-image
fixed test and exported as a separate candidate:

`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r3`

The export contains three annotated images, `detections.csv`, `results.json`,
`manifest.json` and `v0_metadata.json`. The checkpoint SHA-256 is
`40873efeaf5dd2da18a2116bd6147e9b957c1f795bb4b3efeb00404e98c36f30`; camera and
open-world actions remain disabled. GUI tests were rerun after the export:
`32 passed in 6.54s`.

## 当前限制

- v0 Student 固定测试 mAP@0.5 为 `0.263117`，属于探索基线，不宣称达到0.80目标。
- 正式三种子矩阵、正式报告 DOCX/PDF 和发布级 GUI 人工验收仍未完成。
- 所有后续优化必须继续使用同一固定测试协议，并单独记录运行目录和权重哈希。
