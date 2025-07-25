# Image-Guided Depth Completion using LIDAR-Camera Fusion

**Image-Guided Depth Completion** is a real-time deep learning framework that estimates dense depth maps from sparse LIDAR and monocular RGB camera input. This work proposes an efficient encoder-decoder model architecture for use in autonomous vehicles, using early fusion and Residual Up-Projection Blocks (RUB) to reconstruct high-resolution depth.

> 📝 Inspired by "LIDAR and Monocular Camera Fusion: On-road Depth Completion for Autonomous Driving"

---

## 📌 Key Contributions

- **Early Fusion Strategy:** Combine RGB image and sparse depth into a single 4-channel tensor at input
- **Residual Up-Projection Decoder:** Introduces 5×5 and 3×3 convolutions with shortcut connections to preserve edge information
- **BerHu Loss:** Uses a hybrid L1-L2 formulation, better than MSE/MAE for preserving structure
- **Skip Connections:** From encoder layers to decoder for multi-scale feature reuse
- **Lightweight & Real-Time:** Suitable for deployment in real-time embedded systems

---

## 🧠 Architecture Overview
Input (RGB + Sparse LIDAR)
↓
Early Fusion
↓
ResNet-50 Encoder
↓
Skip Connections
↓
Residual Up-Projection Blocks × 4
↓
Conv → Dense Depth Output (1 channel) --- 
---

## ⚙️ Hyperparameters

| Parameter             | Value                    |
|-----------------------|--------------------------|
| Input Size            | 320 × 256                |
| Epochs                | 20                       |
| Batch Size            | 16                       |
| Optimizer             | SGD                      |
| Learning Rate         | 0.01 (Cosine Decay)      |
| Loss Function         | BerHu                    |
| Encoder               | ResNet-50 (pretrained)   |
| Decoder               | 4 × Residual Up-Project  |
| Skip Connections      | Yes                      |
| Fusion Strategy       | Early (Input-Level)      |

---

## 🔬 BerHu Loss Function (ℒ<sub>BerHu</sub>)

The **BerHu** loss (Reverse Huber) combines L1 and L2 norms:

\[
ℒ_{berhu}(e) =
\begin{cases}
|e| & \text{if } |e| \le c \\
\frac{e^2 + c^2}{2c} & \text{if } |e| > c
\end{cases}
\]

Where \( e = y_{pred} - y_{true} \) and \( c = \frac{1}{5} \cdot \max(|e|) \).  
This loss balances robustness and smooth convergence, especially on sparse ground truths.

---

## 📊 Results

### NYUdepthV2

| Method         | Input     | RMSE ↓ | REL ↓ | δ₁ ↑  | δ₂ ↑  | δ₃ ↑  |
|----------------|-----------|--------|--------|--------|--------|--------|
| Eigen et al.   | RGB       | 0.641  | 0.158  | 76.9%  | 95.0%  | 98.8%  |
| Ma et al.      | RGB + D   | 0.230  | 0.044  | 97.1%  | 99.4%  | 99.8%  |
| **Ours**       | RGB + D   | **0.203** | **0.040** | **97.6%** | **99.5%** | **99.9%** |

### KITTI Odometry

| Method         | Input     | RMSE ↓ | REL ↓ | δ₁ ↑  | δ₂ ↑  | δ₃ ↑  |
|----------------|-----------|--------|--------|--------|--------|--------|
| Mancini et al. | RGB       | 7.51   | —      | 31.8%  | 61.7%  | 81.3%  |
| Ma et al.      | RGB + D   | 3.85   | 0.083  | 91.9%  | 97.0%  | 98.9%  |
| **Ours**       | RGB + D   | **3.67** | **0.072** | **92.3%** | **97.3%** | **98.9%** |

---

## 📂 Dataset Structure
DATA/
├── nyudepthv2/
│ ├── images/ # RGB frames
│ ├── depth_sparse/ # Projected LIDAR
│ └── depth_gt/ # Ground truth dense maps
└── kitti
---

## 🛠️ How to Run

```bash
# Clone the repository
git clone https://github.com/Akroy5/Image_Guided_Depth_Completion.git
cd Image_Guided_Depth_Completion

# Setup environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r CODE/requirements.txt

# Run training
python CODE/model.py --data-dir DATA/ --epochs 20 --batch-size 16 EOF
