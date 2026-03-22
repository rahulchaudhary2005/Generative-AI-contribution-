from transformers import pipeline
import torch

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# 🧠 Prompt
# =========================

prompt = PromptTemplate(
    template='Generate 5 interesting things about {topic}',
    input_variables=['topic']
)


# =========================
# ⚡ DEVICE SETUP (AUTO GPU/CPU)
# =========================

device = 0 if torch.cuda.is_available() else -1


# =========================
# 🤖 Create HF Pipeline (GPU ENABLED)
# =========================

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=device,   # ✅ GPU if available
    max_length=512,  # ⚡ reduce for speed
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)


# =========================
# 🤖 Chat Model Wrapper
# =========================

model = ChatHuggingFace(llm=llm)


# =========================
# 🧩 Parser
# =========================

parser = StrOutputParser()


# =========================
# 🔗 Chain
# =========================

chain = prompt | model | parser


# =========================
# ▶️ Run
# =========================

result = chain.invoke({'topic': 'black hole'})
print(result)


# displaying the chain structure
print("\nChain Structure:")
chain.get_graph().print_ascii()