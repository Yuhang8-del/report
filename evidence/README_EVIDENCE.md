# 客户反馈中“两项缺失数据”的说明

截图中的两个红圈不是指模型缺少两种水果，也不是指训练图片丢失，而是要求论文同时提供两组可核查的实验数据：

1. Trust Filter 的伪标签审计证据，以及“94.3%”的具体计算过程；
2. Teacher 和 Student 在全部五个检测类别上的逐类别结果，包括最弱类别。

这些数据原本保存在完整实验 `artifacts` 中。精简运行包最初只保留了推理必需的代码和模型，因此没有复制评估 JSON。当前运行包已补充本目录中的原始证据文件。

## 1. 94.3% 伪标签精度是什么意思

94.3% 是过滤后伪标签在受保护审计集上的 **Precision（精确率）**，不是模型总体准确率，也不是 mAP。

审计集在伪标签筛选前独立封存，包含：

- 45 张真实图片；
- 170 个由人工标注的真实水果框；
- 正确判定规则：预测类别与人工类别相同，并且预测框和真实框的 IoU 不低于 0.50；
- 一对一匹配，避免一个预测框重复匹配多个真实框。

过滤后保留 35 个预测框，其中：

- TP（正确保留）= 33；
- FP（错误保留）= 2；
- FN（未被保留的真实框）= 137。

因此：

```text
Precision = TP / (TP + FP)
          = 33 / (33 + 2)
          = 0.942857
          = 94.3%
```

同时：

```text
Recall = TP / (TP + FN)
       = 33 / (33 + 137)
       = 19.4%
```

这说明 Trust Filter 的策略比较保守：留下来的伪标签大多数正确，但它舍弃了很多较难的真实目标。报告不能只写 94.3%，也必须同时说明 19.4% 的召回率和审计样本数量。

各类别阈值在另一份 90 张图片的固定验证成员上选择，而不是用上述 45 张审计图片调出来：

| 类别 | 置信度阈值 |
|---|---:|
| Apple | 0.549 |
| Banana | 0.850 |
| Orange | 0.850 |
| Strawberry | 0.500 |
| Pineapple | 0.500 |

对应文件：

- `pseudo_label_audit.json`：完整 TP、FP、FN、Precision、Recall、分类别指标和证据哈希；
- `protected_pseudo_audit_labels.json`：受保护审计集的人工标签记录；
- `threshold_calibration_fixed_validation_pr.json`：独立验证成员上的 Precision–Confidence 校准记录；
- `fixed_aspect_ratio_bounds.json`：伪标签几何过滤边界；
- `fixed_split_manifest.json`：固定数据划分及成员关系。

## 2. 全部五类检测结果是什么意思

模型注册的五类水果是 Apple、Banana、Orange、Strawberry 和 Pineapple。客户反馈要求不能只展示表现较好的四类，还必须把最弱的 Strawberry 一并报告。

固定测试集上的 AP@0.5 如下：

| 类别 | Teacher AP@0.5 | Student AP@0.5 |
|---|---:|---:|
| Apple | 0.650 | 0.545 |
| Banana | 0.649 | 0.516 |
| Orange | 0.652 | 0.571 |
| Strawberry | 0.352 | 0.290 |
| Pineapple | 0.840 | 0.740 |

其中 Strawberry 是最弱类别，但它仍然存在于模型权重和类别注册表中。这里所说的“缺失”是指较早版本的报告没有把它放进主要逐类别对比，不代表模型不能输出 Strawberry。

对应文件：

- `teacher_fixed_test.json`：Teacher 总体指标及五类 `per_class_ap50`；
- `student_fixed_test.json`：Student 总体指标、90 张固定测试成员信息及五类 `per_class_ap50`；
- `teacher_confusion_matrix.png`：Teacher 混淆矩阵；
- `student_confusion_matrix.png`：Student 混淆矩阵。

JSON 中类别编号含义如下：

```text
0 = Apple
1 = Banana
2 = Orange
3 = Strawberry
4 = Pineapple
```

## 3. 可以直接回复客户的话

这两个红圈指的是两组实验佐证，并不是缺少两种训练数据。第一组是 94.3% 伪标签精度的审计依据：独立审计集有 45 张图片、170 个真实框，过滤后 33 个正确框、2 个错误框，所以 Precision 为 33/(33+2)=94.3%，同时 Recall 为 19.4%。第二组是五种水果的逐类别测试结果，较早版本没有突出最弱的 Strawberry；模型实际包含 Apple、Banana、Orange、Strawberry 和 Pineapple 五类。现在审计 JSON、五类测试 JSON 和混淆矩阵已经补到运行包的 `evidence` 文件夹中。
