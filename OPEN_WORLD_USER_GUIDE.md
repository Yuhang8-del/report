# 开放世界水果检测使用说明

本项目的开放世界功能分为两层：

1. **11 类显式检测**：原有苹果、香蕉、橙子、草莓、菠萝，加上经确认的牛油果、蓝莓、樱桃、猕猴桃、芒果、网纹瓜。该模型完成后保存为 `models/incremental_11class_best.pt`。
2. **超出 11 类的未知水果发现**：类别无关目标性模型检测尚未被 11 类模型解释的水果框，以黄色虚线标记为 `Unknown`，并显示聚类编号和待人工确认的候选名称。

## 启动演示

在交付根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_open_world_gui.ps1
```

程序会优先加载 `incremental_11class_best.pt`；若该文件尚未生成，则加载五类 `student_best.pt`。未知目标检测还需要以下文件：

- `models/open_world_objectness.pt`
- `models/open_world_encoder.pt`
- `models/open_world_box_clusters.npz`
- `models/open_world_cluster_names.json`

## 图框含义

- 彩色实线框：类别已在当前检测器注册表中，名称是模型的显式类别预测。
- 黄色虚线框：尚未被注册类别解释的水果候选。
- `Cluster Cn / 类别?`：自监督特征聚类后的候选语义，问号表示仍需人工确认，不能当作正式类别标签。

## 评估数据隔离

六个新类使用发现集训练；129 张独立保护测试图（2,884 个框）不进入训练。训练完成后，程序自动在保护集计算总体与逐类 Precision、Recall、mAP@0.5 和 mAP@0.5:0.95，并生成每个新类两张客户效果图。

## 结果目录

- 首版未知框效果图：`outputs/customer_open_world_box_examples/`
- 11 类增量效果图：`outputs/customer_incremental_11class_examples/`
- 11 类真实 GUI 推理截图：`outputs/customer_incremental_11class_gui_screenshots/`
- 11 类保护集指标：`outputs/incremental_11class_protected_metrics.json`

GUI 截图目录包含六个新类别各两张完整窗口截图、汇总图
`gui_inference_contact_sheet.jpg` 和带模型 SHA-256、逐图 SHA-256 的
`gui_screenshot_manifest.json`。截图由真实 PySide6 窗口完成推理后，按
Win32 顶层窗口句柄捕获，不是将离线画框图片后期粘贴到 GUI 模板。

首版未知框基线已经具备真实框级定位能力，但对蓝莓、樱桃等密集小目标漏检较多。最终客户演示应优先使用训练完成后的 11 类增量模型；未知框分支用于继续发现注册表以外的水果。
