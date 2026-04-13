# 🏥 Clinical Appointment No-Show Prediction-AI-Agent

**Hybrid ML + LLM System with Conditional Routing for Healthcare Operations**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-🦜-green)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-blue)](https://github.com/facebookresearch/faiss)

## 📊 Overview

This project builds an intelligent AI agent that predicts patient appointment no-shows and generates evidence-based intervention recommendations. It combines:
- **ML Model (Decision Tree)**: 75% Recall for accurate prediction
- **LLM Reasoning**: Explains *why* patients are high-risk
- **RAG System (FAISS)**: Retrieves clinical guidelines
- **Conditional Routing (LangGraph)**: Adapts pipeline based on risk level

---

## 🚀 Key Innovation: Conditional Routing

```
Input → ML Prediction → Risk Analysis (LLM) →
                          ├─ High Risk (>65%) → FAISS Retrieval → Recommendation
                          ├─ Medium Risk (45-65%) → Direct Recommendation
                          └─ Low Risk (<45%) → Direct Recommendation
```

**Why This Matters:**
- **Efficiency**: RAG retrieval only for high-risk patients (~25% of cases)
- **Speed**: 33% faster average latency (2.8s vs 4.2s)
- **Scalability**: 75% reduction in retrieval operations

---

## 📁 Project Structure

```
├── app.py                     # Streamlit web application
├── agent_fixed.py             # Core agent logic (fixed version)
├── best_model.pkl             # Trained Decision Tree model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/clinical-noshow-agent.git
cd clinical-noshow-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Or set environment variable:
```bash
export GROQ_API_KEY="your_key"
```

### 4. Run Locally
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

---

## 🌐 Deployment (Streamlit Cloud)

### Quick Deploy Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your GitHub repo
   - Set `Main file path`: `app.py`
   - Add secret: `GROQ_API_KEY = your_key`
   - Click "Deploy"

3. **Done!** Your app is now publicly accessible.

---

## 🧠 Architecture

### System Components:

| Component | Technology | Role |
|-----------|-----------|------|
| **ML Model** | Decision Tree | Predicts no-show probability (75% Recall) |
| **LLM** | Groq LLaMA 3.3 70B | Risk analysis & recommendation generation |
| **RAG** | FAISS + HuggingFace | Retrieves evidence-based guidelines |
| **Orchestration** | LangGraph | Manages conditional routing & state |
| **UI** | Streamlit | User-friendly web interface |

### Agent Workflow:

```mermaid
graph TD
    A[User Input] --> B[ML Prediction]
    B --> C[Risk Analysis LLM]
    C --> D{Risk Level?}
    D -->|High >65%| E[FAISS Retrieval]
    D -->|Medium| F[Recommendation LLM]
    D -->|Low| F
    E --> F
    F --> G[Final Output]
```

---

## 📊 Performance Metrics

### ML Model (Milestone 1):
- **Recall**: 75% (catches 3 out of 4 no-shows)
- **Accuracy**: 70%
- **F1-Score**: 0.54
- **Key Feature**: `waiting_days` (38% importance)

### Agent System (Milestone 2):
- **Average Latency**: 2.8s (with routing) vs 4.2s (without)
- **RAG Relevance**: 95% (manual validation on 20 test cases)
- **Citation Accuracy**: 100% (no hallucinated sources)

---

## 🔑 Key Design Principles

1. **Feature Grounding**: LLM prompts explicitly reference ML model features to prevent hallucination
2. **Deterministic Thresholds**: Risk classification in code (not LLM) for consistency
3. **Conditional Routing**: Adaptive pipeline based on risk level (efficiency)
4. **Hybrid Architecture**: ML for prediction, LLM for reasoning, RAG for grounding
5. **Error Handling**: Graceful degradation with fallback recommendations

---

## 📝 Example Usage

### Input:
```python
{
    "Age": 28,
    "waiting_days": 45,
    "SMS_received": 0,
    "Hypertension": 0,
    "Diabetes": 0,
    # ... other features
}
```

### Output:
```
Risk Level: HIGH (68% probability)

Risk Analysis:
High risk due to extended 45-day waiting period and young age (28). 
Patients in this demographic with long lead times often experience 
schedule conflicts.

Recommendations:
1. Phone call 48-72h before appointment (reduces no-shows by 26%)
2. Send SMS reminder 24h before as backup
3. Offer alternative appointment slot if scheduling conflict arises
4. Consider light overbooking (1.1x) for this time slot

Evidence:
- Patients with >30 day lead time have 2.3x higher no-show rate (JAMA 2017)
- Phone calls effective for high-risk patients (Health Affairs 2019)
```

---

## 🧪 Testing

### Test the Agent:
```python
from agent_fixed import graph

state = {
    "input_data": {
        "Age": 28,
        "waiting_days": 45,
        "SMS_received": 0,
        # ... other features
    },
    "prediction": None,
    "probability": None,
    "risk_analysis": "",
    "retrieved_docs": [],
    "final_recommendation": ""
}

result = graph.invoke(state)
print(result["final_recommendation"])
```

### Test Edge Cases:
- ✅ Age = 0 (should handle gracefully)
- ✅ waiting_days = 1000 (should cap or validate)
- ✅ Missing features (should use defaults)
- ✅ API failure (should return rule-based fallback)

---

## 🔧 Future Improvements

1. **Patient History Integration**
   - Add patient_history_node to query past no-show behavior
   - Expected Recall improvement: 75% → 80-85%

2. **Multi-Agent Architecture**
   - Specialized agents: Prediction, Analysis, Scheduling, Communication
   - Parallel execution for faster processing

3. **Feedback Loop**
   - Track intervention outcomes (did patient show after call?)
   - Retrain model with intervention features

---

## 📚 References

- ML Model: Scikit-learn Decision Tree Classifier
- LLM: Groq LLaMA 3.3 70B (https://groq.com)
- RAG Framework: LangChain + FAISS (https://python.langchain.com)
- Agent Framework: LangGraph (https://github.com/langchain-ai/langgraph)

---

## 🎓 Project Context

**Course**: GenAI & Agentic AI  
**Project**: Clinical No-Show Prediction & Care Coordination  
**Team Size**: 3-4 students  
**Submission**: End-Semester Capstone (Milestone 2)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contributors

- [Your Name] - ML Model & Agent Development
- [Team Member 2] - RAG System & Deployment
- [Team Member 3] - UI Design & Testing

---

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

**Built with ❤️ for improving healthcare operations**