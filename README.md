# Gen Ai concepts and all models of langchain

# =========================
# 🧠 1. PROMPTS
# =========================

# Old (still used)
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# New (Recommended)
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


# =========================
# 🤖 2. LLMs & CHAT MODELS
# =========================

from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOllama


# =========================
# 🔗 3. RUNNABLES (NEW)
# =========================

from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

# Old Chains (important for interviews)
from langchain.chains import (
    LLMChain,
    SequentialChain,
    SimpleSequentialChain
)


# =========================
# 🧠 4. MEMORY
# =========================

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)


# =========================
# 📄 5. DOCUMENT LOADERS
# =========================

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    CSVLoader
)


# =========================
# ✂️ 6. TEXT SPLITTERS
# =========================

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)


# =========================
# 🔢 7. EMBEDDINGS
# =========================

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# 🗄️ 8. VECTOR STORES
# =========================

from langchain_community.vectorstores import (
    FAISS,
    Chroma
)

# External
from langchain_pinecone import PineconeVectorStore


# =========================
# 🔍 9. RETRIEVERS
# =========================

from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever
)


# =========================
# 🤖 10. AGENTS
# =========================

from langchain.agents import (
    initialize_agent,
    AgentExecutor,
    create_openai_tools_agent
)


# =========================
# 🔧 11. TOOLS
# =========================

from langchain.tools import Tool
from langchain_core.tools import tool


# =========================
# 🧩 12. OUTPUT PARSERS
# =========================

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)


# =========================
# 🧠 13. MESSAGES / SCHEMA
# =========================

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)


# =========================
# 🔥 MOST IMPORTANT (MUST KNOW)
# =========================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


# =========================
# 🚀 MODERN PIPELINE (NEW STANDARD)
# =========================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "Transformers in NLP"})
print(response)

# =========================
# 🚀 CHAIN FORMATION IN LANGCHAIN
# =========================

