# 水果开放世界框级检测 V1

## 目标

在原有 Apple、Banana、Orange、Strawberry、Pineapple 五类 Student 检测器之外，增加：

1. 类别无关水果框提议；
2. 已知类别框排除；
3. Unknown 框输出；
4. Unknown 框级自监督特征聚类；
5. 候选类别后验映射与人工确认边界；
6. 新类别增量训练接口。

V1 使用可替换的 `UnknownProposalProvider` 接口。当前实现采用仅由五类已知水果训练的一类
Fruit YOLOv8s 提议器，以保证 Windows + RTX 3080 环境可运行；后续可将提议器替换为
PROB/OW-DETR，而不改变评估、聚类和 GUI 接口。

## 数据协议

- 已知目标性训练：8,738 张不重复图片，10,906 次平衡采样输入；所有五类标签折叠为 `Fruit`。
- 已知验证：90 张。
- 未知发现：510 张，类别与检测框在初始训练阶段隐藏。
- 未知受保护测试：129 张。
- 未知类别：Avocado、Blueberry、Cherry、Kiwi、Mango、Rockmelon。
- 训练集与受保护测试集交叉数量：0。

协议目录：

`E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_v1_seed42`

## 运行链路

1. Student 输出五类已知水果框。
2. 一类 Fruit 提议器输出类别无关目标框。
3. 与已知框 IoU 超过阈值的提议被排除。
4. 剩余提议输出为 `Unknown fruit`。
5. 将 Unknown 框裁剪后输入自监督编码器。
6. 使用冻结聚类中心分配 `Unknown Cluster ID`。
7. 候选类别名称只作为后验映射，正式加入类别注册表前必须确认和增量训练。

## 评估指标

- `U-Precision`
- `U-Recall`
- `U-F1`
- `U-AP50`
- `A-OSE`
- 各未知类别召回率
- 框级聚类后验语义准确率

## 类别确认后的增量学习

六个候选类别经确认后，使用 `build_incremental_open_world_dataset` 构建11类数据视图：

- 旧五类回放：3,000张；
- 新六类训练：459张；
- 已知验证：90张；
- 新类验证：51张；
- 受保护未知测试集进入训练的图片：0张；
- 新类别ID按顺序追加为5至10，原五类ID保持不变。

实际数据视图位于：

`E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_incremental_all6_seed42`

## 关键命令

构建协议：

```powershell
$env:PYTHONPATH = 'D:\fruit_ssod_complete_project1\project\src'
python -m fruit_ssod.cli.build_open_world_box_protocol `
  --known-train-list D:\fruit_ssod_complete_project1\data\fruit_ssod\processed\yolo\supervised_v3_domain_balanced_teacher_seed42\train_domain_balanced.txt `
  --known-validation-list D:\fruit_ssod_complete_project1\data\fruit_ssod\processed\yolo\supervised_v3_domain_balanced_teacher_seed42\val.txt `
  --novel-source-root E:\fruit_ssod_runtime\data\fruit_ssod\raw\deepnir\extracted_sanitized\yolov5 `
  --output-root E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_v1_seed42
```

训练类别无关提议器：

```powershell
python -m fruit_ssod.cli.train_open_world_objectness `
  --weights E:\fruit_ssod_runtime\artifacts\weights\yolov8s.pt `
  --dataset E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_v1_seed42\objectness_dataset\dataset.yaml `
  --project E:\fruit_ssod_runtime\artifacts_v18\open_world `
  --name objectness-yolov8s-seed42 --epochs 30 --image-size 640 --batch-size 16
```

完整框级评估：

```powershell
python -m fruit_ssod.cli.evaluate_open_world_boxes `
  --student-weights D:\fruit_ssod_complete_project1\models\student_best.pt `
  --objectness-weights E:\fruit_ssod_runtime\artifacts_v18\open_world\objectness-yolov8s-seed42\weights\best.pt `
  --encoder-checkpoint D:\fruit_ssod_complete_project1\models\open_world_encoder.pt `
  --public-manifest E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\novel_public_manifest.json `
  --protected-truth E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\protected_novel_box_truth.json `
  --output-dir E:\fruit_ssod_runtime\artifacts_v18\open_world\box-evaluation-seed42
```

## 能力边界

- Unknown 框表示模型认为该区域像水果目标，但不属于当前五类。
- Cluster ID 是自监督分组结果，不等同于正式类别 ID。
- 后验候选名称必须经过确认，随后更新类别注册表并执行增量训练。
- V1 的类别无关提议器是工程基线，不宣称达到 PROB 的标准公开基准结果。
