Vector Systems
├── Vector Store (Lightweight)
│   ├── In-Memory Storage
│   ├── Embedding Support
│   ├── Similarity Indexing
│   │   ├── FAISS
│   │   └── HNSW (e.g., via FAISS)
│   ├── Fast Retrieval
│   └── No Persistence / No Metadata Filters
│
└── Vector Database (Full-featured)
    ├── Persistent Storage
    ├── CRUD Operations
    │   ├── Add Vectors
    │   ├── Read by ID / Similarity
    │   ├── Update Vectors
    │   └── Delete Vectors
    ├── Metadata Filtering (e.g., tags, fields)
    ├── Distributed Architecture
    ├── Durability (Backup/Restore)
    ├── Authentication & Authorization
    └── Examples:
        ├── Pinecone
        ├── Weaviate
        ├── Qdrant
        └── Milvus
