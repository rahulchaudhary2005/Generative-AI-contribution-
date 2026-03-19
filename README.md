# GEN AI ALL CONCEPTS AND MODELS ARE WRITTEN THEIR #
# ===================== 🧠 PROMPTS =====================

# Old
from langchain.prompts import (
    PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# New (🔥 Recommended)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder


# ===================== 🤖 MODELS =====================

from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOllama


# ===================== 🔗 RUNNABLES =====================

from langchain_core.runnables import (
    RunnableSequence, RunnableLambda,
    RunnableParallel, RunnablePassthrough
)

# Old (Interview)
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain


# ===================== 🧠 MEMORY =====================

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)


# ===================== 📄 LOADERS =====================

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader,
    DirectoryLoader, CSVLoader
)


# ===================== ✂️ SPLITTERS =====================

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# ===================== 🔢 EMBEDDINGS =====================

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# ===================== 🗄️ VECTOR DB =====================

from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import PineconeVectorStore


# ===================== 🔍 RETRIEVERS =====================

from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever


# ===================== 🤖 AGENTS =====================

from langchain.agents import initialize_agent, AgentExecutor, create_openai_tools_agent


# ===================== 🔧 TOOLS =====================

from langchain.tools import Tool
from langchain_core.tools import tool


# ===================== 🧩 PARSERS =====================

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


# ===================== 🧠 MESSAGES =====================

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ===================== 🔥 CORE (MUST KNOW) =====================

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


# ===================== 🚀 MODERN PIPELINE =====================

prompt = ChatPromptTemplate.from_template("Explain {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
response = chain.invoke({"topic": "Transformers in NLP"})
print(response)
