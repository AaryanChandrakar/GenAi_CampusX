from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2d prompt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n  {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic': 'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'topic':result1.content})
result2 = model.invoke(prompt2)

print(result1.content)
print(result2.content)

