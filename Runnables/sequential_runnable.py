from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import torch


# =========================
# 🤖 Pipeline (GPU/CPU)
# =========================

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=device
)

llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)


# =========================
# 🧠 Prompts
# =========================

prompt1 = PromptTemplate(
    template='Generate 5 interesting things about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Expand on the following points:\n{points}',
    input_variables=['points']
)


# =========================
# 🧩 Parser
# =========================

parser = StrOutputParser()


# =========================
# 🔗 Chain
# =========================

chain = RunnableSequence(
    prompt1, model, parser, prompt2, model, parser
)


# =========================
# ▶️ Run
# =========================

print(chain.invoke({'topic': 'black hole'}))