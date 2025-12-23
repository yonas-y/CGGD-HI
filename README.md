# 🧩 Constraint-Guided Deep Learning for Health Indicator Estimation (CGGD-HI)

**Physically Consistent Health Indicators for Predictive Maintenance and Asset Health Monitoring**

---

## 📌 Project Overview

This repository presents **Constraint-Guided Gradient Descent (CGGD)**, an industry-oriented deep learning approach for **robust health indicator (HI) estimation** in **bearing prognostics and health management (PHM)**.

Traditional data-driven models often achieve high predictive accuracy but fail to enforce **physical plausibility**, while physics-based models struggle with incomplete or uncertain system knowledge.  
This work bridges that gap by **embedding domain constraints directly into the training process**, producing health indicators that are:

- Physically meaningful  
- Monotonic and bounded  
- Robust across operating conditions  
- Suitable for downstream **Remaining Useful Life (RUL)** and maintenance decision systems  

The approach is validated on bearing degradation data and is directly applicable to **industrial predictive maintenance pipelines**.

---

## 🎯 Business & Industrial Relevance

This project addresses common challenges in real-world asset monitoring systems:

- ❌ Unstable or non-interpretable health indicators  
- ❌ Models that violate known degradation behavior  
- ❌ Heavy manual tuning of loss functions  
- ❌ Limited trust in black-box ML outputs  

**CGGD-HI enables:**
- Trustworthy condition indicators for **maintenance planning**
- Better integration of ML models into **industrial workflows**
- Reduced reliance on ad-hoc regularization or heuristic losses

---

## ⚙️ Core Methodology
## ⚙️ Core Methodology

### 📦 Baseline: Convolutional Autoencoder (CAE)

A convolutional autoencoder learns a compact latent representation of time–frequency input data and reconstructs it with minimal error.

The Health Indicator (HI) is defined as:

$$
f_{\mathrm{HI}}^{\mathrm{CAE}}(X) = - \| X - D(E(X)) \|_2
$$

> Higher reconstruction errors correspond to more degraded system states.

---

### 🧠 Constraint-Guided Gradient Descent (CGGD)

Instead of augmenting the loss function with manually weighted penalty terms, CGGD formulates training as a **constrained optimization problem**:

$$
\begin{aligned}
& \min_{\theta} \mathcal{L}_{\mathrm{recon}}(X; \theta) \\
& \text{s.t. } C_i(X; \theta) \le 0, \quad i=1, \dots, M
\end{aligned}
$$

Domain knowledge is encoded as **constraints**, and parameter updates are guided using **constraint-specific gradient directions**.  
This eliminates fragile loss-weight tuning and ensures constraint satisfaction by design.

---

## 🧩 Domain Constraints Implemented

### 🔻 Monotonic Degradation Constraint
The predicted HI is enforced to **decrease monotonically over time**, reflecting inevitable physical wear:

$$
HI_{t+1} \le HI_t
$$

> **Industrial benefit:** Stable degradation trends suitable for maintenance forecasting.

---

### ⚡ Energy–HI Consistency Constraint
If two consecutive samples have similar energy levels, their HIs should also be similar:

$$
| HI_{t+1} - HI_t | \le \epsilon \quad \text{if } | E_{t+1} - E_t | \le \delta
$$

> **Industrial benefit:** Noise-robust indicators under varying operating conditions.

---

### 📏 HI Boundary Constraint
HI values are constrained to a normalized range:

$$
0 \le HI_t \le 1
$$

- **1** → fully healthy  
- **0** → failed  

> **Industrial benefit:** Standardized indicators compatible with dashboards, alarms, and decision rules.

---

### 🔹 CGGD Parameter Update Rule

The parameters are updated as:

$$
\theta_{j+1} := \theta_j - \eta \left( \nabla_\theta \mathcal{L}_{\mathrm{recon}} + \sum_i R_i \, \mathrm{dir}_i \right)
$$

Where:

- $\eta$ → learning rate  
- $R_i$ → constraint activation weight  
- $\mathrm{dir}_i$ → constraint-specific gradient direction  

> This ensures that **all constraints are satisfied** while minimizing the reconstruction loss.

---

## 🧪 Tools & Technology Stack

- **Deep Learning:** TensorFlow / Keras (or PyTorch)  
- **Numerical Computing:** NumPy, SciPy  
- **Visualization:** Matplotlib, Seaborn  
- **MLOps & Experiment Tracking:** ZenML, MLflow  

---

## 🛠️ Getting Started

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

---

## 📈 Typical Use Cases


- Predictive maintenance systems
- Health indicator generation for RUL estimation
- Condition monitoring under limited labeled data
- Trustworthy ML models for industrial assets

---

## 🎯 Why This Repository Matters (For Recruiters)

This repository demonstrates the ability to:

- Translate LLM capabilities into **reliable, structured systems**
- Go beyond prompt engineering to include **tools, memory, and persistence**
- Design **scalable agent architectures** aligned with real product needs
- Apply modern AI frameworks in a **production-oriented manner**

It reflects **practical engineering judgment**, not just experimentation.

---

## 📬 Contact & Collaboration
📧 Email: y[yonas.yehualaeshet@gmail.com](mailto:yonas.yehualaeshet@gmail.com)
🐛 Issues: Open an issue in this repository

