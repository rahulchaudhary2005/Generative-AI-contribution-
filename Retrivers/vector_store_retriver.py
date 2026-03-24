from re import search
from unittest import result

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from torch import embedding

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# create the vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

#convert the vector store to retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query="How can I build applications with LLMs?"
results = retriever.invoke(query)

# print the results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.page_content}")
    print("\n" + "="*50 + "\n")
    
#with vectorstore.similarity_search(query) we can also get the results but with retriever we can also pass search_kwargs to control the number of results and other parameters.

result2=vector_store.similarity_search(query, k=2)
print("Results from similarity_search:")

for i,doc in enumerate(result2):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print("\n" + "="*50 + "\n")