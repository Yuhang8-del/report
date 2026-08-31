# 客户运行版验证记录

## 验证结论

本目录不是代码示例或空壳界面，而是包含真实模型权重的可运行交付包。

- Windows + Conda 环境启动：通过；
- Python 3.10.20：通过；
- PyTorch 2.5.1+cu121：通过；
- NVIDIA GeForce RTX 3080 / CUDA：通过；
- 五类 Student 权重加载与样例推理：通过，样例检出 4 个目标；
- 十一类扩展权重加载与样例推理：通过，样例检出 3 个目标；
- 五类和十一类摄像头模型配置检查：通过；
- 开放类别运行文件完整性检查：通过；
- PySide6 GUI 离屏启动和 Student 权重加载：通过，退出码 0。

## 已验证类别

五类 Student：Apple、Banana、Orange、Strawberry、Pineapple。

十一类扩展模型：Apple、Banana、Orange、Strawberry、Pineapple、Avocado、Blueberry、Cherry、Kiwi、Mango、Rockmelon。

## 关键权重校验

- `student_best.pt` SHA-256：`fef2314905d48028ebbe3d94ea52af222bde7e8255d6475081606e2c309020e4`
- `incremental_11class_best.pt` SHA-256：`ee524c94ef0cac731d8d11dac697ee17ad90700d9138477a80b14a7bd47d249c`

验证输出同时保存在 `outputs/self_check.json`。
