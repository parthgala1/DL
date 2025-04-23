# 🔍 CNN Architecture Summary for Interviews

This guide covers major CNN architectures — AlexNet, VGG, ResNet, InceptionNet, MobileNet, EfficientNet, and custom ConvNets — with detailed flows, activation functions, neurons/filters, and innovations.

---

## 1. 🧠 AlexNet (2012)

> First CNN to dominate ImageNet. Introduced ReLU and Dropout.

### Architecture:

```
Input: 227x227x3
Conv1: 96 filters, 11x11, stride 4 → ReLU → MaxPool (3x3)
Conv2: 256 filters, 5x5, stride 1 → ReLU → MaxPool (3x3)
Conv3: 384 filters, 3x3 → ReLU
Conv4: 384 filters, 3x3 → ReLU
Conv5: 256 filters, 3x3 → ReLU → MaxPool (3x3)
Flatten
FC1: 4096 → ReLU → Dropout
FC2: 4096 → ReLU → Dropout
FC3: 1000 → Softmax
```

- **Activation:** ReLU
- **Innovation:** Dropout, ReLU, Data Augmentation

---

## 2. 🧱 VGGNet (2014)

> Uses deep networks with repeated small (3x3) filters.

### VGG-16 Architecture:

```
Input: 224x224x3
2 x Conv3-64 → MaxPool
2 x Conv3-128 → MaxPool
3 x Conv3-256 → MaxPool
3 x Conv3-512 → MaxPool
3 x Conv3-512 → MaxPool
Flatten → FC-4096 → FC-4096 → FC-1000 → Softmax
```

- **Activation:** ReLU
- **Neurons:** 64 → 128 → 256 → 512
- **Downsampling:** MaxPool after each block

---

## 3. 🪜 ResNet (2015)

> Solves vanishing gradients using **skip connections**.

### ResNet-50 Architecture:

```
Input: 224x224x3
Conv1: 7x7, 64 → MaxPool
Conv2_x: 3 residual blocks (64)
Conv3_x: 4 residual blocks (128)
Conv4_x: 6 residual blocks (256)
Conv5_x: 3 residual blocks (512)
AvgPool → FC-1000 → Softmax
```

- **Block:** F(x) = x + ConvBlock(x)
- **Activation:** ReLU
- **Innovation:** Skip connections, batch norm

---

## 4. 🔀 GoogLeNet / Inception v1 (2014)

> Introduced the **Inception Module** with multi-size filters.

### Inception Module:

```
1x1 Conv
3x3 Conv (after 1x1)
5x5 Conv (after 1x1)
3x3 MaxPool → 1x1 Conv
→ Concatenate all outputs
```

### GoogLeNet:

```
Input → Conv → MaxPool → 9x Inception Modules → AvgPool → FC-1000
```

- **Activation:** ReLU
- **Innovation:** 1x1 bottleneck, no FC at the end

---

## 5. 🧠 Inception-v3 (2015)

> Improves Inception with **factorization** (3x3 → 1x3 + 3x1) and **auxiliary classifiers**.

### Features:

- Factorized convs (to reduce params)
- RMSProp optimizer
- Label smoothing
- Training with auxiliary heads

---

## 6. ⚡ MobileNet (2017)

> Lightweight network for mobile using **depthwise separable convolutions**.

### MobileNet Block:

```
Depthwise Conv (per channel)
→ Pointwise Conv (1x1)
```

### Architecture:

```
Input → Conv 3x3
→ Depthwise + Pointwise (repeated)
→ AvgPool → FC-1000 → Softmax
```

- **Activation:** ReLU6
- **Innovation:** Depthwise separable convs

---

## 7. 📈 EfficientNet (2019)

> Scales network width, depth, and resolution efficiently.

### EfficientNet-B0 Architecture:

```
Stem: Conv 3x3
MBConv1: 3x3, 16
MBConv6: 3x3, 24 → 40 → 80 → 112 → 192 → 320
Head: Conv 1x1 → Pool → FC → Softmax
```

- **Block:** MBConv (inverted residual)
- **Activation:** Swish (x \* sigmoid(x))
- **Innovation:** Compound scaling (depth, width, resolution)

---

## 8. ⚙️ Generic ConvNet (Custom)

> Common CNN setup for tasks like MNIST, CIFAR.

### Example:

```
Input: 28x28x1
Conv: 32 filters, 3x3 → ReLU → MaxPool
Conv: 64 filters, 3x3 → ReLU → MaxPool
Flatten
Dense: 128 → ReLU
Output: 10 → Softmax
```

---

## 📊 Summary Table

| Model        | Key Idea             | Activation | #Params         | Special Layers            |
| ------------ | -------------------- | ---------- | --------------- | ------------------------- |
| AlexNet      | Deep + Dropout       | ReLU       | 60M+            | Dropout, Big kernels      |
| VGG          | Simplicity, deep     | ReLU       | 138M+           | Repeated Conv3 blocks     |
| ResNet       | Residual connections | ReLU       | ~25M (ResNet50) | Skip connections          |
| GoogLeNet    | Inception modules    | ReLU       | ~5M             | 1x1 bottleneck, Inception |
| Inception-v3 | Factorized convs     | ReLU       | ~23M            | 1x3 + 3x1 convs           |
| MobileNet    | Mobile optimized     | ReLU6      | ~4M             | Depthwise Separable Convs |
| EfficientNet | Compound scaling     | Swish      | ~5M+            | MBConv, compound scaling  |
