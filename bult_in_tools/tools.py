# =========================================================
# 🤖 FULL LANGCHAIN AGENT (LOCAL + HUGGINGFACE)
# =========================================================

from transformers import pipeline
import torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType


# =========================================================
# ⚡ DEVICE SETUP
# =========================================================

device = 0 if torch.cuda.is_available() else -1


# =========================================================
# 🤖 LOAD LOCAL MODEL (LIGHTWEIGHT)
# =========================================================

pipe = pipeline(
    "text-generation",
    model="distilgpt2",   # 🔥 lightweight model
    device=device,
    max_length=200
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)


# =========================================================
# 🛠️ DEFINE TOOLS
# =========================================================

# 🔹 Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"


# 🔹 Sentiment Tool
sentiment_pipe = pipeline("sentiment-analysis")

@tool
def sentiment_analysis(text: str) -> str:
    """Analyze sentiment of text."""
    result = sentiment_pipe(text)[0]
    return f"{result['label']} (score: {round(result['score'], 3)})"


# 🔹 Summarization Tool (simple version)
@tool
def summarize(text: str) -> str:
    """Summarize text (basic local logic)."""
    return text[:100] + "..."


# 🔹 Custom Knowledge Tool
@tool
def knowledge_base(query: str) -> str:
    """Answer basic IPL-related questions."""
    data = {
        "kohli": "Virat Kohli is a top IPL batsman and former RCB captain.",
        "dhoni": "MS Dhoni is CSK captain and one of the best finishers.",
        "rohit": "Rohit Sharma has won 5 IPL titles with Mumbai Indians."
    }
    for key in data:
        if key in query.lower():
            return data[key]
    return "No data found."


# =========================================================
# 🧰 REGISTER TOOLS
# =========================================================

tools = [
    calculator,
    sentiment_analysis,
    summarize,
    knowledge_base
]


# =========================================================
# 🤖 CREATE AGENT
# =========================================================

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# =========================================================
# ▶️ RUN AGENT
# =========================================================

if __name__ == "__main__":
    query = "What is the sentiment of this sentence: I love AI and how much is 25 * 4?"

    result = agent.run(query)

    print("\n🤖 Final Answer:\n", result)