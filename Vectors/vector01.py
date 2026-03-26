# =========================================================
# 🧠 LangChain + Chroma Vector DB (IPL Player Knowledge Base)
# =========================================================

# 🔹 Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# =========================================================
# 📄 STEP 1: Create Documents
# =========================================================
# Each Document = text + metadata

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

# Combine all documents
docs = [doc1, doc2, doc3, doc4, doc5]


# =========================================================
# 🔢 STEP 2: Create Embeddings Model
# =========================================================
# Converts text → vectors (numerical representation)

embedding_model = HuggingFaceEmbeddings()


# =========================================================
# 🗄️ STEP 3: Create Chroma Vector Store
# =========================================================
# Stores embeddings + enables similarity search

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory='my_chroma_db',   # saved locally
    collection_name='ipl_players'
)


# =========================================================
# ➕ STEP 4: Add Documents to Vector DB
# =========================================================

vector_store.add_documents(docs)


# =========================================================
# 📊 STEP 5: View Stored Data
# =========================================================

all_data = vector_store.get(include=['documents', 'metadatas'])
print("\n📄 Stored Documents:\n", all_data)


# =========================================================
# 🔍 STEP 6: Similarity Search
# =========================================================
# Find relevant docs based on query

results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2   # top 2 results
)

print("\n🔍 Similarity Search Results:\n", results)


# =========================================================
# 📈 STEP 7: Search with Scores
# =========================================================

results_with_score = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

print("\n📊 Results with Scores:\n", results_with_score)


# =========================================================
# 🎯 STEP 8: Metadata Filtering
# =========================================================
# Filter by team

filtered_results = vector_store.similarity_search(
    query="",
    filter={"team": "Chennai Super Kings"}
)

print("\n🎯 Filtered Results (CSK):\n", filtered_results)


# =========================================================
# 🔄 STEP 9: Update Document
# =========================================================
# NOTE: You must know the document ID beforehand

updated_doc = Document(
    page_content="Virat Kohli is the highest run scorer in IPL history and former RCB captain. Known for consistency, aggression, and elite fitness.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Replace with actual ID from your DB
doc_id = "your-document-id-here"

# Uncomment when you have correct ID
# vector_store.update_document(document_id=doc_id, document=updated_doc)


# =========================================================
# ❌ STEP 10: Delete Document
# =========================================================

# Uncomment when you have correct ID
# vector_store.delete(ids=[doc_id])


# =========================================================
# 📌 FINAL VIEW
# =========================================================

final_data = vector_store.get(include=['documents', 'metadatas'])
print("\n📌 Final DB State:\n", final_data)