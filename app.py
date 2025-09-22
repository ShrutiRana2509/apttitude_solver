import streamlit as st
import re
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate

# ------------------- Streamlit Config -------------------
st.set_page_config(
    page_title="Quizzy – Friendly aptitude assistant",
    page_icon="",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align:center;'> Quizzy – Friendly aptitude assistant</h1>",
    unsafe_allow_html=True
)

# ------------------- Groq LLM -------------------
groq_api_key = "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"
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.0,
    streaming=True,
    max_tokens=1500
)

# ------------------- Hybrid Math & Reasoning -------------------
hybrid_prompt = """
You are an expert in aptitude, math, and logical reasoning. 
Solve the problem **step by step**.

For each step:
1. Identify known quantities (numbers, units).  
2. Convert units if necessary.  
3. Apply correct formulas clearly.  
4. Show intermediate calculations.  
5. Highlight the final numeric answer with correct units.  
6. If multiple solution methods exist, choose the simplest one.

Question: {question}

Answer:
"""

reasoning_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["question"], template=hybrid_prompt)
)

# ------------------- Fast Arithmetic -------------------
def fast_eval(expr: str) -> str:
    try:
        if re.fullmatch(r"[\d\s+\-*/().]+", expr):
            return str(eval(expr))
    except:
        return None
    return None

# ------------------- Session State -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": " Hi! Ask me any aptitude, logic, or math question!"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------- User Input -------------------
question = st.chat_input("Enter your question here...")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("⚡ Solving..."):
        try:
            # Step 1: Fast arithmetic if pure numbers
            arithmetic_result = fast_eval(question)
            if arithmetic_result:
                response = f"**Answer:** {arithmetic_result}"
            else:
                # Step 2: Word problem / reasoning
                response = reasoning_chain.run(question)
        except Exception as e:
            response = f" Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

