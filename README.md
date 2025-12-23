# 🧩 Constraint Guided Gradient Descent Learning for Health Indicator Estimation (CGGD-HI)

---

## 📌 Objective

This work presents a constraint-guided deep learning (\text{DL}) framework to develop physically consistent health indicators (\text{HIs}) in bearing prognostics and health management. Conventional data-driven approaches often lack physical plausibility, while physics-based models are limited by incomplete knowledge of complex systems. To address this, we integrate domain knowledge into \text{DL} models via constraints, ensuring monotonicity, bounding output ranges between 1 and 0 (representing healthy to failed states, respectively), and maintaining consistency between signal energy trends and \text{HI} estimates. Using constraints eliminates the need for complex loss term balancing to incorporate domain knowledge. The constraint-guided gradient descent algorithm (\text{CGGD}) is used to train a \text{DL} model that satisfies specific constraints.

This project explores the use of **Constraint Guided Gradient Descent (CGGD)** for training deep learning models—specifically, convolutional autoencoders (CAE)—to estimate health indicators (HI) for bearings.

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


\subsubsection{Monotonic Degradation Constraint}
When a bearing is put into operation, it undergoes inevitable wear, which progressively worsens with time. This degradation should be reflected in the predicted \text{HI}, which is expected to decrease monotonically over time. To enforce this constraint, the \text{HI} estimates should be penalized if they deviate from the expected monotonic trend based on the time location of the corresponding input data samples.

\subsubsection{Energy-\text{HI} Consistency Constraint}
Building on the monotonic degradation constraint, we expect that while the \text{HI} values should decrease over time, the difference between the \text{HI} values of two consecutive samples should not vary significantly unless a substantial change in signal energy occurs. To enforce this concept, a constraint is introduced that ensures that if two samples have similar energy levels, their \text{HI} estimates should also be close in value. 

\subsubsection{\text{HI} Boundary Constraint}
When predicting the \text{HI} of a bearing, it is convenient that the values remain within a normalized range: a fully healthy state is represented by a value of $ub = 1$, while a failure state corresponds to a value of $lb = 0$. To enforce this condition, boundary constraints are enforced during the training process to ensure that all \text{HI} predictions fall within this defined range. 

---

At the core of the \text{CGGD} optimization procedure, the update of the model parameters is defined in Eq.~(\ref{eq:CGGDUsageSingleColumn}).
\begin{figure*}[htbp]
\hrule
    \begin{align}
        \label{eq:CGGDUsageSingleColumn}
        \theta_{j+1} := \theta_j - \eta \left( \frac{\partial \mathcal{L}_{\text{reconn}}\left(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\mathcal{D}} \right)}{\partial \theta_j} + \right. & \left. \max\left\{ \left\|\nabla_{\mathcal{E}} \mathcal{L}_{\text{reconn}} \left(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\mathcal{D}} \right)\right\|,\epsilon\right\} \frac{\partial f^{\text{CCAE}}_{\text{HI}}\left(\mathcal{E}\left(X_t\right)\right)}{\partial \theta_j} \right. \\ \nonumber
        & \Bigl[R_{\text{mono}} \operatorname{dir}_{\text{mono}} \left(X_t, \bm{X}, \bm{t}, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}}\right) F_{\text{MH}} \left(X_t,\operatorname{dir}_{\text{mono}}(X_t, \bm{X}, \bm{t}, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}})\right) \\ \nonumber 
        & \quad + R_{\text{ene}} \operatorname{dir}_{\text{ene}} \left(X_t, X_{t_0}, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}}\right) F_{\text{MH}} \left(X_t,\operatorname{dir}_{\text{ene}}(X_t, X_{t_0}, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}})\right) \\ \nonumber 
        & \quad + R_{\text{upper}} \operatorname{dir}_{\text{upper}} \left(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}}\right) F_{\text{MH}} \left(X_t,\operatorname{dir}_{\text{upper}}(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}})\right) \\ \nonumber 
        & \quad + R_{\text{lower}} \operatorname{dir}_{\text{lower}} \left(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}}\right) F_{\text{MH}} \left(X_t,\operatorname{dir}_{\text{lower}}(X_t, \bm{\theta}_{\mathcal{E}}, \bm{\theta}_{\text{HI}})\right) \Bigr] \Bigr)
    \end{align}
\hrule
\end{figure*}


## 🧰 Tools & Frameworks
- Python, TensorFlow/Keras (or PyTorch)
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

## 📜 Citation

If you use this work, please cite:
```bash
@article{phm2025cggdhi,
title = {Constraint-Guided Learning of Data-driven Health Indicator Models: An Application on Bearings},
author = {Yonas Tefera, Quinten Van Baelen, Maarten Meire, Stijn Luca and Peter Karsmakers},
journal = {Vol. 16 No. 2 (2025): International Journal of Prognostics and Health Management },
year = {2025},
}
```
---

## 📬 Contact
Questions, suggestions, or contributions?  
Open an issue or contact: [yonas.yehualaeshet@gmail.com](mailto:yonas.yehualaeshet@gmail.com)
