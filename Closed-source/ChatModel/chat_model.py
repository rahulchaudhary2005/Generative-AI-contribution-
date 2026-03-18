from json import load

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model= ChatOpenAI(model='gpt-4')
ans=model.invoke("What is the capital of France?")
print(ans)

# if u want to print the contet only then use ans.content
print(ans.content)