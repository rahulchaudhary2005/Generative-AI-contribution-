# Generative-AI-contribution-
Starting up the Generative ai learning path 


##🧠 🔥 1. PROMPTS MODULE##
📦 Old (still widely used)
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
📦 New (Recommended 🔥)
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
🤖 🔥 2. LLMs & CHAT MODELS
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatOllama
🔗 🔥 3. CHAINS / RUNNABLES (NEW WAY)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)
⚠️ Old Chains (still asked in interviews)
from langchain.chains import (
    LLMChain,
    SequentialChain,
    SimpleSequentialChain
)
🧠 4. MEMORY
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
📄 5. DOCUMENT LOADERS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    CSVLoader
)
✂️ 6. TEXT SPLITTERS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
🔢 7. EMBEDDINGS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
🗄️ 8. VECTOR STORES
from langchain_community.vectorstores import (
    FAISS,
    Chroma
)

👉 External:

from langchain_pinecone import PineconeVectorStore
🔍 9. RETRIEVERS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever
)
🤖 10. AGENTS
from langchain.agents import (
    initialize_agent,
    AgentExecutor,
    create_openai_tools_agent
)
🔧 11. TOOLS
from langchain.tools import Tool
from langchain_core.tools import tool
🧩 12. OUTPUT PARSERS
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
🧠 13. SCHEMA (MESSAGES)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
🔥 MOST IMPORTANT (Don’t Miss These)

If you're short on time, master these imports first:

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
🚀 PRO TIP (Senior-Level Insight)

LangChain is moving towards:
👉 langchain_core + runnable pipeline

Example modern pipeline:

prompt = ChatPromptTemplate.from_template("Explain {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
