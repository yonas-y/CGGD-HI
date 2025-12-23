# 🧩 Constraint-Guided Deep Learning for Health Indicator Estimation (CGGD-HI)
### Physically Consistent Health Indicators for Predictive Maintenance and Asset Health Monitoring

## 📌 Project Overview
This repository presents **Constraint-Guided Gradient Descent (CGGD)**, an industry-oriented deep learning approach for **robust health indicator (HI) estimation in bearing prognostics and health management (PHM)**.

Traditional data-driven models often achieve high predictive accuracy but fail to enforce **physical plausibility**, while physics-based models struggle with incomplete or uncertain system knowledge.
This work bridges that gap by **embedding domain constraints directly into the training process**, producing health indicators that are:
- Physically meaningful
- Monotonic and bounded
- Robust across operating conditions
- Suitable for downstream **Remaining Useful Life (RUL)** and maintenance decision systems

The approach is validated on bearing degradation data and is directly applicable to **industrial predictive maintenance pipelines**.

---

## 🚀 What This Repository Demonstrates

- Building **LLM-powered agents** with Google ADK  
- Tool-augmented reasoning and action execution  
- **Structured and schema-validated outputs**  
- **Stateful and persistent agents**  
- **Multi-agent coordination and delegation**  
- Clean, modular, and extensible agent architectures  

---

## 🧩 Example Gallery

### 1️⃣ Basic Agent  
📁 **Location:** `1-basic-agent/`

A minimal agent illustrating core ADK setup and interaction.

**Key concepts:**
- Agent initialization
- Prompt and instruction design
- Model selection fundamentals

---

### 2️⃣ Tool-Enabled Agent  
📁 **Location:** `2-tool-agent/`

An agent capable of invoking external tools (functions or APIs) to augment its reasoning.

**Key concepts:**
- Tool integration with `FunctionTool`
- Agent–tool interfaces
- Action-oriented agent design

---

### 3️⃣ LiteLLM-Backed Agent  
📁 **Location:** `3-litellm-agent/`

An agent using **LiteLLM** to interface with alternative LLM providers.

**Key concepts:**
- Multi-provider LLM integration
- Vendor-agnostic agent design
- Cost- and flexibility-aware deployment

---

### 4️⃣ Structured Output Agent  
📁 **Location:** `4-structured-outputs/`

An agent that produces **schema-validated structured outputs** using Pydantic.

**Key concepts:**
- JSON output enforcement
- Output validation
- Reliable downstream system integration

---

### 5️⃣ Sessions & Stateful Interaction  
📁 **Location:** `5-sessions-and-state/`

An agent that maintains conversational state and user context across interactions.

**Key concepts:**
- Session services
- Context persistence
- Personalization foundations

---

### 6️⃣ Persistent Storage Agent  
📁 **Location:** `6-persistent-storage/`

A reminder agent backed by **SQLite-based persistent storage**.

**Key concepts:**
- Database-backed agent memory
- CRUD operations
- Durable state across restarts

---

### 7️⃣ Multi-Agent Manager  
📁 **Location:** `7-multi-agent/`

A manager agent coordinating multiple specialized sub-agents (e.g., news analysis, stock analysis, trend prediction).

**Key concepts:**
- Multi-agent orchestration
- Task delegation
- Compositional agent architectures

---

## 🛠️ Getting Started

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Configure Environment Variables
Create a `.env` file (per example, if required) and add your API keys and configuration values.

### 3️⃣ Run an Example
```bash
cd 6-persistent-storage/reminder_agent
python main.py
```

---

## 🧪 Technology Stack

- **Google Agent Development Kit (ADK)**
- **Google Generative AI**
- **LiteLLM**
- **Pydantic**
- **SQLite**
- **Python**
- `yfinance`, `psutil`, `python-dotenv`

---

## 🎯 Why This Repository Matters (For Recruiters)

This repository demonstrates the ability to:

- Translate LLM capabilities into **reliable, structured systems**
- Go beyond prompt engineering to include **tools, memory, and persistence**
- Design **scalable agent architectures** aligned with real product needs
- Apply modern AI frameworks in a **production-oriented manner**

It reflects **practical engineering judgment**, not just experimentation.

---

## ✅ Best Practices Followed

- Modular and readable code structure
- Clear separation of agent logic, tools, and storage
- Environment-based configuration for secrets
- Reusable patterns suitable for production adaptation

---

## 📄 License

Licensed under the **Apache 2.0 License** — suitable for learning, experimentation, and extension.
