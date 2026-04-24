from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Write a brief information on {topic}',
    input_variables = ['topic']
)

report_prompt = PromptTemplate(
    template = 'Write a concise summary on {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt | model | parser | report_prompt | model | parser

result = chain.invoke({'topic':'Madhya Pradesh and Chhattisgarh Partition.'})

print(result)

chain.get_graph().print_ascii()
