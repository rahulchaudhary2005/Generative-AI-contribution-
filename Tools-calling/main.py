# ================================
# LangChain + HuggingFace Offline Tools Agent
# ================================

import os
import math
import platform
from transformers import pipeline
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import HuggingFacePipeline

# ================================
# 1. LOAD OFFLINE HUGGINGFACE MODEL
# ================================

# Use a lightweight offline model (download once)
generator = pipeline(
    "text-generation",
    model="distilgpt2",   # lightweight (works offline after first download)
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=generator)

# ================================
# 2. DEFINE REAL-WORLD TOOLS
# ================================

# Tool 1: Calculator
def calculator_tool(query: str):
    try:
        return str(eval(query))
    except:
        return "Error in calculation"

# Tool 2: File Reader
def file_reader_tool(filename: str):
    try:
        with open(filename, "r") as f:
            return f.read()
    except:
        return "File not found"

# Tool 3: System Info
def system_info_tool(_):
    return f"OS: {platform.system()}, Processor: {platform.processor()}"

# Tool 4: Text Summarizer (HuggingFace)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_tool(text: str):
    try:
        result = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except:
        return "Error in summarization"

# Tool 5: Word Count
def word_count_tool(text: str):
    return f"Word count: {len(text.split())}"

# ================================
# 3. REGISTER TOOLS
# ================================

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Performs mathematical calculations. Input should be a math expression."
    ),
    Tool(
        name="File Reader",
        func=file_reader_tool,
        description="Reads content from a file. Input should be filename."
    ),
    Tool(
        name="System Info",
        func=system_info_tool,
        description="Returns system OS and processor info."
    ),
    Tool(
        name="Summarizer",
        func=summarize_tool,
        description="Summarizes long text."
    ),
    Tool(
        name="Word Counter",
        func=word_count_tool,
        description="Counts words in given text."
    )
]

# ================================
# 4. INITIALIZE AGENT
# ================================

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ================================
# 5. RUN EXAMPLES (TOOL CALLING)
# ================================

if __name__ == "__main__":

    print("\n--- Calculator Example ---")
    print(agent.run("What is 45 * 12 + 10?"))

    print("\n--- System Info Example ---")
    print(agent.run("Give me my system information"))

    print("\n--- Word Count Example ---")
    print(agent.run("Count words in: LangChain is powerful framework"))

    print("\n--- Summarization Example ---")
    text = """Artificial Intelligence is transforming industries by enabling machines to learn from data,
    make decisions, and automate processes. It is widely used in healthcare, finance, and transportation."""
    print(agent.run(f"Summarize this text: {text}"))

    print("\n--- File Reader Example ---")
    print(agent.run("Read file sample.txt"))