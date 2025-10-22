# 🃏 Card AI - Recognizing Playing Cards with PyTorch

Card AI is a computer vision project that uses **PyTorch** and a **ResNet18** model to recognize all playing cards in a standard deck. The model is trained to classify each card (52 or 54 with jokers) using deep learning techniques.

---

## ⚡ Requirements

Before running the project, install the required dependencies:

```bash
pip install torch torchvision matplotlib
```

---

## 🛠️ Libraries

| Library       | Purpose                                        | Installation              |
| ------------- | ---------------------------------------------- | ------------------------- |
| `torch`       | Neural network framework and model training    | `pip install torch`       |
| `torchvision` | Pre-trained models and image transformations   | `pip install torchvision` |
| `matplotlib`  | Visualization of images and results (optional) | `pip install matplotlib`  |

---

## 📁 Project Structure

Recommended directory layout for training and evaluation:

```
card-ai/
├─ data/
│  ├─ train/
│  │  ├─ AS_of_spades/...
│  │  ├─ 2_of_spades/...
│  │  └─ ...
│  └─ val/
├─ models/
├─ notebooks/
├─ train.py
├─ dataset.py
└─ README.md
```

---

## 🧠 Model Overview

* The model is based on **ResNet18**, available from `torchvision.models`.
* The final fully connected layer is replaced to output the number of card classes (e.g., 52).
* You can start with a **pre-trained** ResNet18 for faster convergence.
* Data augmentation with `torchvision.transforms` helps improve generalization.

Example:

```python
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 52)  # 52 classes for standard deck
```

---

## ⚙️ Training Tips

* Use **transfer learning**: freeze early layers and fine-tune the final ones.
* Apply data augmentations like random crop, resize, flip, and normalization.
* Start with a low **learning rate** (e.g., 0.001) and use schedulers like `StepLR`.
* Use `CrossEntropyLoss` for classification.
* Always monitor validation accuracy to prevent overfitting.

---

## 📊 Visualization

You can visualize sample predictions or training progress using Matplotlib:

```python
import matplotlib.pyplot as plt
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Predicted: {pred_label}")
plt.show()
```

---

## ✅ Next Steps

* Add `train.py` script for full model training.
* Implement a custom `dataset.py` to load card images.
* Experiment with fine-tuning, learning rate schedules, and more augmentations.

---

**Author:** samuelc-01dev

