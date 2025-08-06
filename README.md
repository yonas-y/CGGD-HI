# 🧩 Constraint Guided Gradient Descent for Health Indicator Learning (CGGD-HI)

> Deep learning–based health indicator modeling using a **Constraint Guided Gradient Descent (CGGD)** framework and convolutional autoencoders (CAE).  
> Learn meaningful, monotonic, and interpretable health indicators directly from raw data.

---

## 📌 Overview

This project explores the use of **Constraint Guided Gradient Descent (CGGD)** for training deep learning models—specifically, convolutional autoencoders (CAE)—to estimate health indicators (HI) for rotating machinery (e.g., bearings).

---

## ⚙️ How It Works

### 📦 Baseline: Convolutional Autoencoder (CAE)
- Learn a **compact latent representation** (z) of input time–frequency data (X)
- Reconstruct (X) from (z) with minimal loss
- Health Indicator (HI) defined as the negative reconstruction error:

    f^{CAE}_{HI}(X) = -||X - D(E(X))||_2

Higher reconstruction errors (lower HI) typically correspond to faulty states.

---

### 🧠 CGGD: Adding Constraints
We reformulate the CAE training objective as a **constrained optimization problem**:

    minimize   L_reconn(X, θ_E, θ_D)
    subject to C_i(X, θ_E, θ_D) ≤ 0, for i=1,...,M

Constraints (C_i) capture domain knowledge (e.g., smoothness, monotonicity).  
Custom **constraint directions** guide updates toward solutions satisfying these constraints.

---

### 🔧 CCAE: Learnable HI
To increase flexibility:
- Instead of computing HI directly from reconstruction error,
- Define HI as a **learnable function** of the latent encoding:

    f^{CCAE}_{HI}(E(X))

where θ_HI are the parameters of the HI model.

---

## 📊 Input Data
- Acceleration signals from bearings
- Transformed into **time–frequency representation** X ∈ ℝ^{D × T}
- Used as input to the model

---

## 🧰 Tools & Frameworks
- Python, PyTorch (or TensorFlow/Keras)
- MLOps: [ZenML](https://zenml.io/), [MLflow](https://mlflow.org/)
- Data processing: NumPy, SciPy
- Visualization: Matplotlib, Seaborn

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/cggd-hi.git
cd cggd-hi
pip install -r requirements.txt
```

---

## 🚀 Getting Started

```python
from model import CCAE

# Initialize and train your constrained autoencoder
model = CCAE()
model.fit(train_data)
```

More examples will be available in the [examples](./examples) folder.

---

## 📜 Citation

If you use this work, please cite:
> "Constraint Guided Learning of a Deep Learning based Health Indicator Model"  
> *Yonas Tefera*, et al.

---

## 📬 Contact
Questions, suggestions, or contributions?  
Open an issue or contact: [yonas.yehualaeshet@gmail.com](mailto:yonas.yehualaeshet@gmail.com)
