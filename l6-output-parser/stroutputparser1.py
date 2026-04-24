from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2d prompt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary report on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)
