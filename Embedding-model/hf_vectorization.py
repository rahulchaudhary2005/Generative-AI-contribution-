from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India. It is a bustling metropolis with a rich history and vibrant culture. The city is known for its iconic landmarks such as the Red Fort, India Gate, and Qutub Minar. Delhi is also famous for its diverse cuisine, bustling markets, and vibrant arts scene. It serves as the political, cultural, and economic hub of the country, attracting millions of visitors each year."
vector=embeddings.embed_query(text)
print(vector)
