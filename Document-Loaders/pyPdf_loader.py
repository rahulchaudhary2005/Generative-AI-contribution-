from pydoc import doc

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs={"max_length": 2048, "temperature": 0.7}
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

parser=StrOutputParser()
loader=PyPDFLoader("./Rahul_chaudhary_AIML_resume.pdf")
documents=loader.load()
print(documents[0].metadata)