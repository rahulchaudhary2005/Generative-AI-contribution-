from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import PromptTemplate
import wikipedia

retriver=WikipediaRetriever(top_k_results=2, lang='en')

query='The geopolitical history of india and pakistan '
doc=retriver.invoke(query)

#printing the docs one by one 
for i, d in enumerate(doc):
    print(f"Document {i+1}:")
    print(f"Title: {d.metadata['title']}")
    print(f"Content: {d.page_content[:500]}...")  # Print the first 500 characters of the content
    print("\n" + "="*50 + "\n")