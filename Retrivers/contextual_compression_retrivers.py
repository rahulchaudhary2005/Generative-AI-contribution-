# Contextual_compression_retrivers in python huggingface langchain

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMChainExtractoror
from langchain_core.documents import Document
from sklearn import base

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]


# Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

base_retriver = vector_store.as_retriever(search_kwargs={"k": 3})

#setup the compressor

llm= ChatHuggingFace(model_name="gpt2")
compressor=LLMChainExtractoror.from_llm(llm=llm,
                                        input_template="Extract the most relevant sentence from the following document for the query: {query}\nDocument: {document}")

#create the conterxtual compreaainson retriver
contextual_compression_retriver=ContextualCompressionRetriever(
                                 base_retriever=base_retriver,
                                 compressor=compressor
                                 )

# Query the retriever
query = "What is photosynthesis?"
compressed_results = contextual_compression_retriver.invoke(query)

# Print the compressed results
for i, doc in enumerate(compressed_results):
    print(f"Compressed Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print("\n" + "="*50 + "\n")
