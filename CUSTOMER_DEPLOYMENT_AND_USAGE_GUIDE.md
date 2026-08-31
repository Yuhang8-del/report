# 半监督水果检测项目——客户本地部署与使用清单（详细版）

## 0. 先阅读这一页

本清单用于客户在自己的 Windows 电脑上部署、启动、验证和使用本项目。请按顺序操作，不要跳过“环境安装”和“自检”两步。

交付目录可以放在任意本地磁盘；以下将其统称为“交付根目录”。当前示例使用 `D:\2`，如果客户电脑上使用的是其他位置，请把命令中的 `D:\2` 改成实际位置。日常 GUI 启动脚本会自动识别交付根目录，不依赖原开发机的 E 盘或共享盘。

客户日常部署验收只需要完成第 1 至第 10 节。第 11 节“完整训练归档恢复”仅供科研复现使用，耗时长、占用空间大，不是演示前的必做步骤。

## 1. 客户接收后先确认文件完整

### 1.1 不要只复制 `project` 文件夹

客户必须复制整个交付根目录。根目录应同时包含：

| 必需项目 | 位置 | 作用 |
|---|---|---|
| 安装脚本 | `setup_environment.ps1` | 创建 Python 运行环境 |
| 自检脚本 | `self_check.ps1` | 验证模型、CUDA、样例推理和开放类别文件 |
| 主 GUI | `run_gui.ps1` | 启动英文 GUI |
| 开放类别 GUI | `run_open_world_gui.ps1` | 启动离线 Unknown 候选分析 |
| 模型 | `models` | 五类、十一类、Teacher 和开放类别模型 |
| 样例 | `samples\images` | 部署后第一批测试图片 |
| 项目代码 | `project` | 源代码、依赖和启动逻辑 |
| 实验证据 | `artifacts\v17` | 已完成实验的评估、日志和伪标签证据 |
| 最终报告 | `reports\final_report` | PDF、Word、数据图表 |
| 完整归档 | `archives` | 完整训练数据及完整历史产物 |

当前完整交付约为 **55 GB**。仅用于 GUI 推理和演示，目标磁盘至少应保留 **70 GB** 可用空间；如果要解压完整训练数据并重新训练，建议保留 **200 GB 或更多**可用空间。

### 1.2 推荐的放置方式

1. 将完整交付文件夹复制到本机 NTFS 本地磁盘，例如 `D:\2` 或 `D:\fruit_ssod_project`。
2. 不要把它放在 OneDrive、百度网盘同步目录、U 盘、网络映射盘或只读目录。
3. 不要在外层再多套一层同名目录。打开根目录后应能直接看到 `run_gui.ps1`、`models` 和 `project`。
4. 复制完成后，在资源管理器中确认 `models\student_best.pt` 和 `models\incremental_11class_best.pt` 都存在。

在 PowerShell 中可执行以下命令进行最小文件检查：

```powershell
$DeliveryRoot = 'D:\2'              # 改为客户实际交付目录
Set-Location $DeliveryRoot
Test-Path .\run_gui.ps1
Test-Path .\models\student_best.pt
Test-Path .\models\incremental_11class_best.pt
Test-Path .\samples\images
```

四行结果都应为 `True`。如任意一项为 `False`，说明目录位置不对或复制不完整，先重新复制，再继续。

## 2. 客户电脑准备条件

| 项目 | 最低要求 | 推荐/已验证环境 |
|---|---|---|
| 操作系统 | Windows 10/11，64 位 | Windows 原生 PowerShell |
| 内存 | 16 GB | 32 GB |
| 显卡 | 静态图片可 CPU 推理，但很慢 | NVIDIA RTX 3080（10 GB 显存）已验证 |
| 驱动 | `nvidia-smi` 可正常运行 | 支持 CUDA 12.1 的 NVIDIA 驱动 |
| Python 环境 | Conda + Python 3.10 | 独立 `fruit-ssod` Conda 环境 |
| 网络 | 首次安装依赖时需要 | 可访问 PyPI 与 `download.pytorch.org` |
| 摄像头 | 非必需 | USB 摄像头，Windows 允许桌面应用访问 |

本项目不要求客户单独安装 CUDA Toolkit。安装脚本会通过锁定依赖安装 CUDA 版 PyTorch；驱动正常、`nvidia-smi` 能运行即可。

### 2.1 先检查 NVIDIA 显卡

在 PowerShell 执行：

```powershell
nvidia-smi
```

正常时会显示显卡名称、驱动版本、显存占用等信息。若提示找不到 `nvidia-smi`，请先由客户 IT 或 NVIDIA 驱动安装程序修复显卡驱动，再部署本项目。不要以 CPU 模式替代实时摄像头或训练验收。

## 3. 安装 Miniconda（客户机器未安装 Conda 时）

如果执行 `conda --version` 已能看到版本号，请跳到第 4 节。

### 3.1 下载和安装

1. 用浏览器打开 Anaconda 官方下载页：

```text
https://www.anaconda.com/download/success
```

2. 在 Windows 区域选择 **Miniconda Windows 64-Bit Graphical Installer**。Miniconda 只提供 Conda、Python 和必要依赖，适合本项目；不需要安装包含大量预装包的 Anaconda Distribution。
3. 双击安装包，选择 **Just Me**。
4. 安装目录建议保持默认的用户目录，例如 `C:\Users\<用户名>\miniconda3`；不要安装到项目目录内。
5. 保持默认选项完成安装。无需勾选“加入系统 PATH”；脚本会自动查找常见 Miniconda/Anaconda 安装位置。
6. 从开始菜单打开 **Miniconda Prompt**，执行：

```powershell
conda init powershell
```

7. 关闭所有 PowerShell 窗口，再打开一个新的 Windows PowerShell。

### 3.2 验证 Conda

在新 PowerShell 中执行：

```powershell
conda --version
conda info --envs
```

第一行应输出类似 `conda 25.x` 的版本信息。出现版本号即表示 Conda 安装完成；不要求此时已经有 `fruit-ssod` 环境。

## 4. 创建项目运行环境

### 4.1 打开正确的 PowerShell 位置

最简单的方法是在交付根目录的空白处按住 Shift 并右键，选择“在此处打开 PowerShell”。也可以逐条执行：

```powershell
$DeliveryRoot = 'D:\2'              # 改为客户实际交付目录
Set-Location $DeliveryRoot
Get-Location
```

最后一行应显示客户的交付根目录。确认后执行安装命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_environment.ps1 -EnvName fruit-ssod
```

### 4.2 安装过程中脚本会做什么

脚本将依次：

1. 找到已安装的 Conda；
2. 创建名为 `fruit-ssod` 的 Python 3.10 环境；如果环境已存在则复用；
3. 下载并安装锁定版本的 PyTorch、Ultralytics、OpenCV、PySide6 等依赖；
4. 安装项目自身代码；
5. 隔离用户目录中其他 Python 包，避免 Qt/PySide 冲突；
6. 自动运行 `pip check`，检查依赖冲突。

首次下载可能需要较长时间。安装期间请保持网络连接，不要关闭窗口，不要额外执行 `pip install torch` 或在 base 环境安装包。

### 4.3 什么算安装成功

安装完成后，控制台最后应出现：

```text
Environment setup completed. Run run_gui.ps1 from the delivery root.
```

接着执行：

```powershell
conda env list
conda run -n fruit-ssod python --version
conda run -n fruit-ssod python -m pip check
```

应能看到 `fruit-ssod` 环境、Python 3.10.x，以及 `No broken requirements found.`。三项均正常后再进入自检。

## 5. 运行部署自检（必须完成）

在交付根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\self_check.ps1 -EnvName fruit-ssod
```

### 5.1 自检会验证什么

| 自检项 | 客户应看到的结果 |
|---|---|
| Python | Python 3.10.x |
| PyTorch | `2.5.1+cu121` |
| CUDA | NVIDIA 机器应显示 `"cuda_available": true` |
| 五类 Student | Apple、Banana、Orange、Strawberry、Pineapple |
| 十一类扩展模型 | 上述五类外还有 Avocado、Blueberry、Cherry、Kiwi、Mango、Rockmelon |
| 开放类别文件 | objectness、encoder、clusters、names 四个文件均存在 |
| 样例推理 | 五类 Student 与十一类模型均返回检测结果 |
| 摄像头配置 | 输出 `Semi-Supervised Student (5 Classes)` 和 `Extended Detector (11 Classes)` |

### 5.2 自检成功的标志

输出中必须出现：

```text
"status": "passed"
[通过] 自检结果已写入：...\outputs\self_check.json
```

请打开 `outputs\self_check.json` 保存到客户验收记录中。该文件记录当前 Python、PyTorch、显卡、模型 SHA-256、类别列表和样例推理数量。

如自检不通过，不要启动 GUI，应先查看第 13 节的故障处理。

## 6. 启动英文 GUI

自检通过后，在交付根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gui.ps1 -EnvName fruit-ssod
```

### 6.1 GUI 打开后先做的三件事

1. 等待主窗口标题 `Fruit Detection & Open-Category Research Studio` 出现。
2. 等待默认五类 Student 模型自动加载完成；窗口顶部的模型提示会从“未加载”变为已加载状态。
3. 如果要用于图片、批量或视频的十一类检测，点击顶部 **Load Model**，选择：

```text
models\incremental_11class_best.pt
```

请勿手动重命名任何 `.pt` 文件。若要回到半监督五类模型，重新通过 **Load Model** 选择：

```text
models\student_best.pt
```

### 6.2 GUI 页签对应关系

| 页签 | 作用 | 建议首次测试 |
|---|---|---|
| Live Camera | USB 摄像头实时检测 | 有摄像头时测试 |
| Image Detection | 单张图片检测 | 必做 |
| Batch Processing | 文件夹或多张图批量检测 | 建议测试 |
| Video File | 本地视频检测并导出 MP4 | 有视频时测试 |
| Experiment Overview | 查看项目流程、类别和系统能力 | 用于客户演示 |

## 7. 图片、批量和视频的具体操作

### 7.1 单张图片检测（建议作为第一次 GUI 测试）

1. 点击左侧 **Image Detection**。
2. 确认模型已加载；若顶部仍显示未加载，点击 **Load Model** 并选择 `models\student_best.pt`。
3. 点击 **Select Images...**。
4. 打开交付目录下的 `samples\images`，任选一张 `.jpg` 或 `.png` 图片。
5. 点击 **Run Detection**。
6. 等待结果画面出现检测框、英文水果名称、置信度和统计信息。
7. 如需保存，点击 **Export Results...**，选择一个新的客户输出目录，例如 `D:\fruit_demo_output`。

单张检测成功的最低标准是：界面不报错，至少能显示输入图与处理结果；不同图片的检测数量可能不同。

### 7.2 批量图片检测

1. 点击 **Batch Processing**。
2. 点击 **Select Folder...** 选择客户自己的图片文件夹，或点击 **Select Images...** 多选图片。
3. 点击 **Run Detection**。
4. 等待进度完成后，用 **Previous / Next** 浏览结果。
5. 点击 **Export Results...**，选择一个空白或新建的输出文件夹。

建议客户的原始图片与检测输出使用不同目录，避免覆盖原图。可支持 JPG、JPEG、PNG 和 BMP。

### 7.3 本地视频检测

1. 点击 **Video File**。
2. 点击 **Open Video...**，选择本地 MP4、AVI 或 MOV 视频。
3. 点击 **Set Output File...**，选择一个新的 `.mp4` 输出位置。
4. 点击 **Start Processing**。
5. 处理中可点击 **Pause** 暂停；需要终止时点击 **Stop**。
6. 处理完成后，打开刚才设置的输出 MP4，确认视频中有检测框与类别信息。

视频过大或编码不兼容时，先用常见播放器确认其可播放，再转换为标准 H.264 MP4 后重试。

## 8. USB 摄像头实时识别的具体操作

### 8.1 启动前准备

1. 将 USB 摄像头连接到电脑。
2. 在 Windows 打开“设置 → 隐私和安全性 → 相机”。
3. 打开“允许桌面应用访问相机”。
4. 关闭系统相机、腾讯会议、Teams、Zoom、浏览器网页摄像头等可能占用设备的程序。

### 8.2 在 GUI 中启动

1. 启动主 GUI，点击左侧 **Live Camera**。
2. 在 **Model** 下拉框选择：
   - `Semi-Supervised Student (5 Classes)`：苹果、香蕉、橙子、草莓、菠萝；
   - `Extended Detector (11 Classes)`：上述五类加牛油果、蓝莓、樱桃、猕猴桃、芒果、网纹瓜/哈密瓜。
3. 在 **Device** 区域点击 **Refresh**。
4. 从下拉列表选择要使用的摄像头编号。
5. 在 **Resolution** 中优先选 `640` 或相近分辨率。
6. 推荐初始设置：置信度 `0.25`，NMS IoU `0.45`。
7. 点击 **Start Detection**。
8. 确认实时画面中显示全部检测框、英文类别名、置信度、FPS、推理耗时及目标数量。
9. 需要保存效果图时，点击 **Save Current Frame**，选择保存位置。默认建议文件名为 `fruit_camera_snapshot.png`。
10. 演示结束点击 **Stop**，状态应由运行中变回断开；此步骤用于释放摄像头。

### 8.3 真实水果现场演示建议

1. 保证正面光线充足，避免逆光。
2. 水果应完整露出，先单个或少量摆放，再测试多个水果同框。
3. 背景尽量简洁，水果占画面高度至少约五分之一。
4. 首先用 `0.25` 置信度；漏检时可降到 `0.20`，误检明显时可升到 `0.30–0.40`。
5. 真实场景与公开训练图片在光照、相机色彩、角度和背景上可能不同，现场表现应以实机测试为准。

## 9. 十一类扩展与离线开放类别功能

### 9.1 十一类扩展检测

十一类是正式的固定类别框级检测。类别为：

```text
Apple, Banana, Orange, Strawberry, Pineapple,
Avocado, Blueberry, Cherry, Kiwi, Mango, Rockmelon
```

使用方式：摄像头页在 Model 下拉框直接选 `Extended Detector (11 Classes)`；图片、批量和视频页则通过顶部 **Load Model** 加载 `models\incremental_11class_best.pt`。

### 9.2 离线开放类别 GUI

在交付根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_open_world_gui.ps1 -EnvName fruit-ssod
```

该程序会优先使用十一类模型。图中：

| 图框 | 含义 |
|---|---|
| 彩色实线框 | 当前模型中已注册的固定类别 |
| 黄色虚线 Unknown | 未被当前已知类别解释的水果候选 |
| Cluster Cn / 类别? | 自监督特征聚类得到的候选建议，需要人工确认 |

请向客户明确：Unknown/Cluster 是离线开放类别实验的候选发现结果，不能直接当作已经确认的新品种名称，也不在实时摄像头逐帧链路中运行。

## 10. 报告与结果证据查看

客户无需重新训练即可查看已经完成的结果。关键位置如下：

| 内容 | 文件 |
|---|---|
| 最终报告 PDF | `reports\final_report\final_report.pdf` |
| 可编辑报告 Word | `reports\final_report\final_report.docx` |
| 高保真报告 Word | `reports\final_report\final_report_high_fidelity.docx` |
| Teacher 固定测试 | `artifacts\v17\runs\supervised-v3-domain-balanced-yolov8m-1024-seed42-r3\evaluations\test.json` |
| Student 固定测试 | `artifacts\v17\runs\ssod-v3-teacher-r3-student-seed42\evaluations\test.json` |
| 伪标签审计 | `artifacts\v17\pseudo\v3_teacher_r3_seed42\audit_report\pseudo_audit.json` |
| 开放类别聚类 | `artifacts\v17\open_world\post_student_ssod-v3-teacher-r3-student-seed42\discovery_results.json` |
| 十一类保护集评估 | `outputs\incremental_11class_protected_metrics.json` |
| 本机部署自检结果 | `outputs\self_check.json` |

JSON 文件中留下的原开发机 E 盘路径是历史来源记录；客户日常部署只通过当前交付根目录的脚本和相对文件完成，不受这些历史路径影响。

## 11. 可选：恢复完整训练数据和历史产物

### 11.1 什么时候需要做

仅在客户需要审计完整原始数据、查看全部历史训练记录或重新开展科研训练时执行。本步骤不是 GUI 推理、摄像头展示、报告查看或客户验收的前置条件。

### 11.2 先校验归档文件

在交付根目录执行：

```powershell
Get-FileHash .\archives\full_training_data.tar.zst -Algorithm SHA256
Get-FileHash .\archives\full_v17_artifacts.tar.zst -Algorithm SHA256
```

对应值必须分别为：

```text
full_training_data.tar.zst
76e09ced833a609a78f867f03576d20f8b0fb08589b7de3392a48c78380ab0ef

full_v17_artifacts.tar.zst
7d025a38f281a9aed778a5ca9542a0f9519ac3b768b60f77582be49290f3fc3a
```

若哈希不同，停止解压，重新复制归档，不使用可能损坏的数据。

### 11.3 解压到独立目录

不要覆盖当前精简版的 `data` 或 `artifacts`。执行：

```powershell
New-Item -ItemType Directory -Force -Path .\restored\data,.\restored\artifacts
tar --zstd -xf .\archives\full_training_data.tar.zst -C .\restored\data
tar --zstd -xf .\archives\full_v17_artifacts.tar.zst -C .\restored\artifacts
```

完成后应存在：

```text
restored\data\fruit_ssod
restored\artifacts\v17
```

如 Windows 自带 `tar` 不支持 `--zstd`，请安装支持 Zstandard 的新版 tar 或 7-Zip 后再解压。

### 11.4 关于完整重新训练

完整 Teacher—伪标签—Student—开放类别训练属于科研复现实验：需要完整归档、10 GB 以上显存、较长运行时间，并会生成新的权重和日志。不同驱动、显卡、依赖版本和随机数实现会造成小幅数值波动。

如需严格执行该链，先阅读：

```text
project\docs\handoff\reproduction.md
project\configs\experiments
```

重新训练必须输出到新的目录，绝不能覆盖交付的 `models`、`artifacts\v17`、固定测试证据或 `archives`。

## 12. 常见故障与按步骤处理

| 问题 | 先做什么 | 仍未解决时 |
|---|---|---|
| `conda` 找不到 | 从开始菜单打开 Miniconda Prompt，执行 `conda init powershell`，关闭并重开 PowerShell | 确认 Miniconda 是否安装在当前用户目录 |
| PowerShell 提示禁止运行脚本 | 必须使用本清单中的 `-ExecutionPolicy Bypass` 完整命令 | 不要修改全局执行策略 |
| 安装下载失败 | 检查浏览器是否能打开 PyPI 与 Anaconda 官方下载页，确认代理/网络 | 重新执行 `setup_environment.ps1`，脚本可复用环境 |
| `pip check` 报错 | 不在 base 环境执行，确认使用 `fruit-ssod` | 环境损坏时，先联系技术支持再删除并重建环境 |
| `cuda_available` 为 False | 执行 `nvidia-smi`，确认驱动能识别 NVIDIA GPU | 更新显卡驱动后重新运行安装和自检 |
| GUI 无法打开 | 先完整运行 `self_check.ps1` | 确认从根目录运行 `run_gui.ps1`，不要双击 Python 文件 |
| 摄像头没有设备 | 检查 Windows 相机权限，关闭占用程序，拔插后点 Refresh | 用 Windows 相机应用确认硬件是否正常 |
| 摄像头画面慢 | 将分辨率设为 640，关闭其他占用 GPU 的程序 | 使用 NVIDIA GPU，不建议 CPU 实时检测 |
| 视频无法读取 | 先确认视频能在本机播放器播放 | 转为标准 H.264 MP4 再试 |
| Unknown GUI 报文件缺失 | 再运行 `self_check.ps1` 检查四个开放类别文件 | 核对 `models` 目录是否复制完整 |

如需要重新创建环境，以下命令会删除指定 Conda 环境；只有在确认环境损坏、且客户允许重建时才执行：

```powershell
conda env remove -n fruit-ssod
```

删除后回到第 4 节，重新运行安装脚本。

## 13. 客户验收记录

| 验收项 | 通过 | 备注 |
|---|---|---|
| 整个交付根目录复制完成 | [ ] |  |
| `conda --version` 能执行 | [ ] |  |
| `fruit-ssod` 环境创建完成 | [ ] |  |
| `pip check` 显示无依赖冲突 | [ ] |  |
| `self_check.ps1` 输出 `status: passed` | [ ] |  |
| CUDA/NVIDIA GPU 可用 | [ ] |  |
| 单张样例图片检测通过 | [ ] |  |
| 批量图片检测并导出通过 | [ ] |  |
| 视频处理并导出通过 | [ ] |  |
| 五类 Student 模型加载通过 | [ ] |  |
| 十一类扩展模型加载通过 | [ ] |  |
| 摄像头刷新、启动、截图、停止通过 | [ ] |  |
| 离线开放类别 GUI 能启动 | [ ] |  |
| 最终 PDF/Word 报告能打开 | [ ] |  |
| `outputs\self_check.json` 已保留 | [ ] |  |

以上必做项均通过后，可确认客户已完成本地部署、模型推理和演示复现。
