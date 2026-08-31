# 水果半监督目标检测演示系统使用说明

## 启动程序

在交付根目录 `D:\fruit_ssod_complete_project1` 打开 Windows PowerShell，执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gui.ps1 -EnvName fruit-ssod
```

首次部署请先运行 `setup_environment.ps1`，随后运行 `self_check.ps1` 检查 Conda、CUDA、模型和样例推理。

## 外接摄像头实时识别

1. 接入 USB 摄像头，并在 Windows“设置 → 隐私和安全性 → 相机”中允许桌面应用访问。
2. 进入“Live Camera”，点击“Refresh”，选择摄像头编号和采集分辨率。
3. 根据演示目的选择模型：

   - 半监督 Student（5类）：苹果、香蕉、橙子、草莓、菠萝。
   - 扩展检测模型（11类）：在上述五类外增加牛油果、蓝莓、樱桃、猕猴桃、芒果、哈密瓜。

4. 设置置信度和 NMS IoU，点击“Start Detection”。界面会绘制每个检测框，并显示英文类别、置信度、FPS、推理耗时、目标数量和类别汇总。
5. 点击“Save Current Frame”可导出带检测框的 PNG；点击“Stop”会释放摄像头，之后可切换设备或模型。

如未发现设备，请关闭 Windows 相机、会议软件等可能占用摄像头的程序，检查系统相机权限后重新刷新。

## 图片、批量与视频演示

1. 点击顶部“加载模型”，默认选择交付目录下的 `models\student_best.pt`。
2. “Image Detection”用于单图检测；“Batch Processing”用于多张图片或文件夹；“Video File”用于本地视频逐帧检测。
3. 推荐样例位于 `samples\images`，推理输出位于 `outputs`。
4. “Experiment Overview”展示 Teacher、Student、半监督流程与十一类扩展能力。

## 功能边界

Teacher/Student 完成五类水果框级半监督检测，十一类增量模型完成扩展水果框级检测。离线开放类别模块还可对五类之外的候选图片进行自监督特征提取、聚类和后验映射，但该 Unknown 分析计算量较大，不在摄像头逐帧链路中运行。GUI 不会把未登记物体强行显示成“未知水果”。

## 报告文件

最终 Word/PDF 报告、统计图和数据位于交付根目录的 `reports\final_report`，原始 Word/PPT 要求文件位于 `requirements`。
