import streamlit as st
import re
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# ------------------- Streamlit APP CONFIG -------------------
st.set_page_config(
    page_title="LogicBot – Solves logical and aptitude questions",
    page_icon="🧮",
    layout="wide"
)
st.markdown("<h1 style='text-align:center;'>LogicBot – Solves logical and aptitude questions</h1>", unsafe_allow_html=True)

# ------------------- Groq API Key -------------------
groq_api_key = "Your_api_key"  # <-- replace with your Groq API key

# ------------------- Initialize LLM -------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0)

# ------------------- Initialize Tools -------------------
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search the Internet for information on various topics."
)

# Math calculator with numeric-only output
math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

def safe_calculator(query: str) -> str:
    """Wrapper to ensure only numeric result is returned from LLMMathChain."""
    try:
        result = math_chain.run(query)
        if isinstance(result, str):
            # Extract only first numeric value
            match = re.search(r"[-+]?\d*\.?\d+", result)
            if match:
                return match.group(0)
        return result
    except Exception as e:
        return f"Calculation error: {e}"

calculator = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Solve math-related questions; only input mathematical expressions."
)

# Reasoning / step-by-step explanation tool
prompt = """
You are an agent tasked with solving the user's mathematical or logical questions. 
Provide a detailed, logical solution in a step-by-step, point-wise format.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Solve logic-based and reasoning questions."
)

# ------------------- Initialize Agent -------------------
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ------------------- Session State -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm your Math & Knowledge chatbot. Ask me anything!"}
    ]

# ------------------- Display Chat -------------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------- User Input -------------------
question = st.chat_input("Enter your question here...")

if question:
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Generating response..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Pass only latest user input
        response = assistant_agent.run(question, callbacks=[st_cb])

        # Append assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
