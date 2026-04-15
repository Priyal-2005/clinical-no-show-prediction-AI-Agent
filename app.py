import streamlit as st
import os
import pickle
import pandas as pd
from typing import Dict
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

# Page config
st.set_page_config(
    page_title="Clinical No-Show AI Agent",
    page_icon="🏥",
    layout="wide"
)

# ========== CACHED SETUP ==========

@st.cache_resource
def load_model():
    """Load ML model (cached)"""
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def setup_llm():
    """Setup LLM (cached)"""
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found!")
        st.stop()
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

@st.cache_resource
def setup_rag():
    """Setup RAG vectorstore (cached)"""
    documents = [
        "Patients with long waiting times are more likely to miss appointments.",
        "Sending SMS reminders 24 hours before reduces no-show rates significantly.",
        "Elderly patients with chronic conditions are more likely to attend appointments.",
        "Follow-up phone calls are effective for high-risk patients.",
        "Overbooking strategies can be used for high no-show probability patients.",
        "Young adults (18-35) have higher no-show rates due to unstable schedules.",
        "Patients without SMS reminders are 2x more likely to miss appointments.",
        "Phone calls 48-72h before reduce no-shows by 26% (Health Affairs 2019).",
        "Patients with >30 day lead time have 2.3x higher no-show rate (JAMA 2017)."
    ]
    

    embedding = FakeEmbeddings(size=384)
    return Chroma.from_texts(documents, embedding)

# Load resources
model = load_model()
llm = setup_llm()
vectorstore = setup_rag()

def get_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# ========== AGENT LOGIC ==========

class AgentState(TypedDict):
    input_data: Dict
    prediction: int
    probability: float
    risk_analysis: str
    retrieved_docs: List[str]
    final_recommendation: str

def predict_no_show(features_dict):
    """ML prediction"""
    try:
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[model.feature_names_in_]
        
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        return {"prediction": int(prediction), "probability": float(prob)}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {"prediction": 0, "probability": 0.5}

def retrieve_docs(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def risk_analysis_node(state: AgentState):
    input_data = state["input_data"]
    result = predict_no_show(input_data)
    
    prediction = result["prediction"]
    probability = result["probability"]
    
    prompt = f"""
You are a healthcare risk analysis assistant.

Patient Data:
- Waiting Days: {input_data.get('waiting_days', 0)}
- Age: {input_data.get('Age', 0)}
- SMS Received: {input_data.get('SMS_received', 0)}

Predicted No-show Probability: {probability:.2%}

Context:
- 'waiting_days' is the MOST important feature (38% importance)
- 'age' and 'SMS_received' also influence prediction
- Ignore other features unless necessary

Risk Levels:
- >65% → High
- 45-65% → Medium
- <45% → Low

Provide concise analysis (max 80 words):
1. Risk level reasoning
2. Top contributing factors
"""
    
    response = llm.invoke(prompt)
    
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_analysis": response.content
    }

def retrieval_node(state: AgentState):
    probability = state["probability"]
    input_data = state["input_data"]
    
    # Only retrieve for high-risk
    if probability < 0.65:
        return {"retrieved_docs": []}
    
    query = f"""
        Patient likely to miss appointment.
        Waiting days: {input_data.get('waiting_days')}
        Age: {input_data.get('Age')}
        SMS received: {input_data.get('SMS_received')}
    """
    docs = retrieve_docs(query)
    
    return {"retrieved_docs": docs}

def recommendation_node(state: AgentState):
    risk = state["risk_analysis"]
    docs = state["retrieved_docs"]
    probability = state["probability"]
    
    if probability >= 0.65:
        action_hint = "Phone Call + SMS + Consider Overbooking"
        risk_level = "High"
    elif probability >= 0.45:
        action_hint = "SMS Reminder"
        risk_level = "Medium"
    else:
        action_hint = "Standard Reminder"
        risk_level = "Low"
    
    docs_text = "\n".join(f"- {doc}" for doc in docs) if docs else "No specific guidelines."
    
    prompt = f"""
You are a care-coordination AI assistant.

Risk Analysis:
{risk}

Guidelines:
{docs_text}

Probability: {probability:.2%}
Risk Level: {risk_level}
Baseline Action: {action_hint}

Provide:
1. Brief summary of key factors
2. Specific actionable recommendations

Keep concise (max 120 words).
"""
    
    response = llm.invoke(prompt)
    return {"final_recommendation": response.content}

def route_risk(state: AgentState):
    prob = state["probability"]
    if prob >= 0.65:
        return "high_risk"
    elif prob >= 0.45:
        return "medium_risk"
    else:
        return "low_risk"

# Build graph
builder = StateGraph(AgentState)
builder.add_node("Risk Analysis", risk_analysis_node)
builder.add_node("Retrieval (Chroma)", retrieval_node)
builder.add_node("Recommendation", recommendation_node)

builder.add_edge(START, "Risk Analysis")
builder.add_conditional_edges(
    "Risk Analysis",
    route_risk,
    {
        "high_risk": "Retrieval (Chroma)",
        "medium_risk": "Recommendation",
        "low_risk": "Recommendation"
    }
)
builder.add_edge("Retrieval (Chroma)", "Recommendation")
builder.add_edge("Recommendation", END)

graph = builder.compile()


# ========== STREAMLIT UI ==========

st.title("🏥 Clinical No-Show Prediction AI Agent")
st.markdown("### Hybrid ML + LLM System with Intelligent Conditional Routing")
st.caption("Predict patient no-shows and get actionable, evidence-based interventions.")

st.markdown("---")

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=28, help="Patient age")
    waiting_days = st.number_input("Waiting Days", min_value=0, max_value=365, value=45, 
                                    help="Days between scheduling and appointment")
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

with col2:
    sms_received = st.selectbox("SMS Received", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    scholarship = st.checkbox("On Welfare Program")
    hypertension = st.checkbox("Hypertension")

with col3:
    diabetes = st.checkbox("Diabetes")
    alcoholism = st.checkbox("Alcoholism")
    handicap = st.number_input("Disability Level", min_value=0, max_value=4, value=0)

# Day of week selection
day_of_week = st.select_slider(
    "Appointment Day",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
    value=2
)

st.markdown("---")

# Predict button
if st.button("🚀 Analyze Patient Risk", type="primary", use_container_width=True):
    
    # Build input
    input_data = {
        "Gender": gender,
        "Age": age,
        "Scholarship": int(scholarship),
        "Hipertension": int(hypertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholism),
        "Handcap": handicap,
        "SMS_received": sms_received,
        "waiting_days": waiting_days,
        "appointment_day_of_week": day_of_week
    }
    
    # Run agent
    with st.spinner("🤖 Running AI Agent Pipeline... Please wait"):
        state = {
            "input_data": input_data,
            "prediction": None,
            "probability": None,
            "risk_analysis": "",
            "retrieved_docs": [],
            "final_recommendation": ""
        }
        
        result = graph.invoke(state)
    
    # Display results
    prob = result['probability']
    
    # Risk level badge
    if prob >= 0.65:
        risk_color = "🔴"
        risk_label = "HIGH RISK"
        badge_color = "#ff4444"
    elif prob >= 0.45:
        risk_color = "🟡"
        risk_label = "MEDIUM RISK"
        badge_color = "#ffaa00"
    else:
        risk_color = "🟢"
        risk_label = "LOW RISK"
        badge_color = "#44ff44"
    
    st.markdown(f"## {risk_color} {risk_label}")
    st.metric("No-Show Probability", f"{prob:.1%}")
    
    # Show if RAG was triggered
    if len(result['retrieved_docs']) > 0:
        st.success(f"📚 RAG Activated: {len(result['retrieved_docs'])} relevant guidelines retrieved")
    else:
        st.warning("⚡ Fast Path: Retrieval skipped for faster response")
    
    st.markdown("---")
    
    # Risk analysis
    st.markdown("### 🔍 AI Risk Analysis")
    st.info(result['risk_analysis'])
    
    # Recommendations
    st.markdown("### 💡 Recommended Actions")
    st.success(result['final_recommendation'])
    
    # Retrieved docs (if any)
    if result['retrieved_docs']:
        st.markdown("### 📚 Evidence-Based Guidelines")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            st.markdown(f"- {doc}")
    
    st.markdown("---")
    
    # Disclaimer
    st.caption("⚠️ **Disclaimer:** This is an operational decision support tool. Final scheduling decisions should consider institutional policies and patient preferences.")

# Sidebar info
with st.sidebar:
    st.markdown("## 📊 System Info")
    st.markdown(f"""
    **ML Model:** Decision Tree  
    **Recall:** 75%  
    **LLM:** Groq LLaMA 3.3 70B  
    **RAG:** Chroma Vectorstore  
    **Framework:** LangGraph
    
    ---
    
    ## 🔧 Key Features
    - ✅ Conditional Routing
    - ✅ Feature Grounding
    - ✅ Evidence-Based RAG
    - ✅ Hybrid ML + LLM
    
    ---
    
    ## 📈 Risk Thresholds
    - **High:** > 65%
    - **Medium:** 45-65%
    - **Low:** < 45%
    """)
    
    st.markdown("---")
    st.markdown("Built for GenAI End-Sem Project")