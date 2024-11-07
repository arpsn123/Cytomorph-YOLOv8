# Microscopic Cellular Image Segmentation with YOLOv8
This project focuses on instance segmentation of microscopic cellular images using YOLOv8, a cutting-edge deep learning model known for its speed and accuracy in object detection and segmentation tasks. Leveraging YOLOv8, this project successfully segments cellular structures of various shapes and sizes, achieving remarkable precision across different cell types. This README provides an in-depth description of the dataset, YOLOv8 architecture, annotations, training process, evaluation metrics, and the tech stack involved.

---

## Table of Contents
1. [Dataset](#dataset)
2. [YOLOv8 Overview](#yolov8-overview)
3. [Annotations](#annotations)
4. [Training Details](#training-details)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Technology Stack](#technology-stack)
7. [How to Run](#how-to-run)

---

## Dataset

### Overview
The dataset, named **Diverse**, consists of **10,000 microscopic cellular images**. This dataset is unique due to its wide variety of cell shapes and types, including square, circular, and triangular cells. This diversity is essential for training the model to recognize and segment a broad spectrum of cellular structures, making it more generalizable.

- **Training set**: 9,000 images
- **Validation set**: 500 images
- **Test set**: 500 images

The dataset is annotated in a format compatible with YOLOv8 and supports various segmentation classes. Each image contains multiple instances, with annotations covering boundaries of each cellular structure, which allows for precise localization and segmentation of individual cells.

---

## YOLOv8 Overview

YOLOv8 is a modern object detection and segmentation framework from the YOLO (You Only Look Once) family, known for its balance between speed and accuracy. YOLOv8 introduces several architectural enhancements that make it highly suitable for real-time segmentation tasks:

### Key Features
1. **Unified Architecture**: YOLOv8 combines both object detection and segmentation in a single framework, enhancing model versatility.
2. **Dynamic Anchor Selection**: Automatically adjusts anchors to fit the shapes and sizes within a dataset, critical for handling diverse cellular shapes.
3. **Decoupled Head**: YOLOv8 separates the detection and segmentation heads, allowing for more refined feature extraction, crucial for segmenting small cellular structures.
4. **Data Augmentation**: YOLOv8 includes built-in data augmentation techniques such as scaling, flipping, and rotation, improving generalization over diverse cellular structures.

---

## Annotations

Annotations for this project were done to support YOLOv8’s requirements for both bounding boxes and segmentation masks. Key aspects of the annotation format include:

1. **Bounding Boxes**: Each cell is annotated with bounding boxes for the detection task. The boxes provide spatial location data for YOLOv8’s detection head.
2. **Segmentation Masks**: Each cellular instance has a polygonal mask that highlights its exact shape. This mask enables YOLOv8’s segmentation head to output accurate cell boundaries.

Each image in the Diverse dataset contains these annotations to facilitate multi-class segmentation, allowing the model to distinguish between different cellular shapes.

---

## Training Details

### Training Configuration
Training was conducted using a high-performance GPU environment with YOLOv8 configured for instance segmentation. The training process involved fine-tuning the model over 50 epochs, with detailed tracking of loss metrics, precision, and recall.

- **Epochs**: 50
- **Batch Size**: Variable, optimized based on GPU capacity
- **Image Size**: 800 pixels
- **Learning Rate**: Optimized through scheduler to reduce oscillations

### Hardware Used
Training utilized a GPU with **8.54 GB** memory, ensuring fast computation and allowing for larger batch sizes.

### Sample Metrics (Epoch 38/50)
At the 38th epoch, the model achieved excellent performance metrics:

```
Epoch       GPU_mem    box_loss    seg_loss    cls_loss    dfl_loss   Instances    Size
38/50        8.54G      0.7109      0.9734       0.397      0.9016      427        800
```

This epoch highlighted the model’s efficient learning curve, with steady improvements across detection and segmentation tasks.

---

## Evaluation Metrics

Upon evaluation at the 38th epoch, YOLOv8 achieved **phenomenal results** on the test set. Below are the precision, recall, and mAP metrics, essential for understanding model performance:

| Metric         | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
|----------------|---------------|------------|--------|-----------|
| Box            | 0.957         | 0.967      | 0.983  | 0.879     |
| Mask           | 0.951         | 0.960      | 0.975  | 0.817     |

These results underscore YOLOv8’s effectiveness in handling various cell shapes and sizes, demonstrating high precision and recall rates for both bounding boxes and masks.

---

## Technology Stack


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-v8.0.0-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5.1-yellow)
![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange)

The following technologies were utilized in this project:
- **Python**: Core language for model implementation and dataset processing.
- **YOLOv8**: Advanced version of the YOLO framework, enabling real-time object detection and segmentation.
- **PyTorch**: Deep learning library used as the backbone for YOLOv8.
- **CUDA**: GPU acceleration framework, essential for training large datasets efficiently.
- **OpenCV**: Used for image preprocessing and augmentation.
- **Matplotlib**: Visualization library for plotting training metrics and results.
- **Jupyter Notebooks**: Environment for iterative experimentation and visualization.

Each tool was selected to ensure streamlined development, efficient training, and easy customization for further model improvement.

---

## How to Run

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU with at least 8 GB memory
- PyTorch 1.8+
- YOLOv8 library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the dataset path in the YOLOv8 configuration file:
   ```yaml
   # config.yaml
   dataset_path: "path/to/diverse_dataset"
   ```

### Training
To train the model on the Diverse dataset:
```bash
python train.py --data config.yaml --epochs 50 --img-size 800
```

### Evaluation
To evaluate model performance:
```bash
python evaluate.py --data config.yaml --weights best.pt
```

---

## Conclusion

The YOLOv8 model showcased remarkable performance in segmenting diverse cellular images, handling various shapes with high precision and recall. This project exemplifies the strength of YOLOv8 in complex instance segmentation tasks, highlighting its applicability in fields requiring detailed cellular analysis.

For further details on customization or deployment, refer to the documentation in the `docs/` folder.