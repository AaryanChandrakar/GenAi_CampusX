from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # Error: No module named 'langchain.output_parsers'

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

schema = [
    ResponseSchema(name='fact1', description='A fact 1 about the topic'),
    ResponseSchema(name='fact2', description='A fact 2 about the topic'),
    ResponseSchema(name='fact3', description='A fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'football'})

print(result)