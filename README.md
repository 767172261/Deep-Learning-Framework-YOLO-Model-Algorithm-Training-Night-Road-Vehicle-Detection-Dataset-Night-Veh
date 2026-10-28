深度学习框架YOLO模型算法训练夜间道路车辆检测数据集 黑夜夜间车辆检测数据集的训练及应用 推理识别检测夜间状态下的车辆
@[toc]

道路夜间车辆检测数据集 yolo8 夜间车辆数据集  31379张  labelimg标注
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cfecc6e1d34b445a9081c73e7cc0f619.png)
**数据集描述：
夜间车辆检测数据集 yolo8 夜间车辆数据集共计31379张  labelimg标注
标签名：car
标注格式：labelimg
格式：txt（yolo）
标签与图片一一对应
已划分train和val** 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1cbbff5161d4cf2a7f415647e62e72b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8917e9215cd04d3087a593d5e55c3fb3.png)
# 🌙 夜间车辆检测数据集 YOLOv8 完整训练与部署

---

## ✅ 一、环境搭建

### 1. 安装 CUDA 驱动（适用于 NVIDIA GPU）

确保你的系统已经安装了支持 CUDA 的 NVIDIA 显卡驱动。使用以下命令查看驱动版本：

```bash
nvidia-smi
```

如果没有安装，请前往 [NVIDIA 驱动下载页面](https://www.nvidia.com/Download/index.aspx) 下载并安装。

---

### 2. 安装 Anaconda

如果你尚未安装 Anaconda，可以从 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载并安装。

创建一个用于 YOLOv8 的 Python 虚拟环境：

```bash
conda create -n night_car_detection python=3.9
conda activate night_car_detection
```

---

### 3. 安装必要依赖项

在激活的虚拟环境中安装以下依赖：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy labelimg
```

---

## 📁 二、数据集结构说明

你的夜间车辆检测数据集包含 **31379 张图像**，仅有一个类别：

- 类别：`car`
- 标注工具：`labelImg`
- 标注格式：YOLOv5/8 支持的 `.txt` 格式
- 已划分为 `train` 和 `val` 集合

### 数据集结构应如下：

```
night_car_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### `data.yaml` 内容示例：

```yaml
train: ./night_car_dataset/images/train
val: ./night_car_dataset/images/val

nc: 1
names: ['car']
```

---

## 🧠 三、YOLOv8 模型训练流程

### 1. 下载 YOLOv8 官方预训练模型（推荐）

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 2. 开始训练（Python API）

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(
    data='night_car_dataset/data.yaml',  # 替换为你的 data.yaml 路径
    epochs=100,
    imgsz=640,
    batch=32,              # 根据显存调整，建议从16或32开始
    device=0,              # 使用 GPU 0
    project='runs/night_car',  # 结果保存路径
    name='exp',            # 实验名称
    save=True,
    save_period=10,        # 每10个epoch保存一次
    pretrained=True,
    verbose=True
)
```

### 或使用命令行方式训练：

```bash
yolo task=detect mode=train model=yolov8s.pt data=night_car_dataset/data.yaml epochs=100 imgsz=640 batch=32 device=0
```

---

## 🔍 四、模型推理（Inference）

### 1. 单张图像推理

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型权重
model = YOLO('runs/night_car/exp/weights/best.pt')

# 推理图像
img_path = 'night_car_dataset/images/val/night_car_0001.jpg'
results = model(img_path)

# 可视化结果
for r in results:
    im_array = r.plot()
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection Result', im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 视频流推理（本地视频或摄像头）

```python
cap = cv2.VideoCapture('night_video.mp4')  # 或使用 0 表示摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('Night Car Detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 五、模型评估（Validation）

使用验证集评估模型性能：

```python
from ultralytics import YOLO

# 加载最佳权重
model = YOLO('runs/night_car/exp/weights/best.pt')

# 执行验证
metrics = model.val(data='night_car_dataset/data.yaml')

# 打印关键指标
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

---

## 📈 六、结果可视化与分析

YOLOv8 会自动将以下内容保存到 `runs/night_car/exp/val/` 目录：

- `confusion_matrix.png`：混淆矩阵
- `precision_recall_curve.png`：PR曲线
- `results.csv`：各 epoch 的损失和指标

你也可以使用 OpenCV 或 Matplotlib 自定义绘制这些图表。

---

## 🧩 七、系统部署建议

### 1. 构建 Web API 服务（可选）

使用 `FastAPI` 或 `Flask` 提供 HTTP 接口，用于上传图像并返回检测结果。

### 2. 部署到边缘设备（如 NVIDIA Jetson）

- 安装 JetPack SDK
- 使用 TensorRT 加速推理
- 部署模型用于车载夜间监控系统

---

## 🛠 八、优化建议

### 1. 数据增强（建议开启）

```python
model.train(..., augment=True)
```

以上文字和仅供参考学习使用
