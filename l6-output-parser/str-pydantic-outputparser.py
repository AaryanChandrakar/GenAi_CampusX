from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define structure using Pydantic
class Facts(BaseModel):
    fact1: str
    fact2: str
    fact3: str

# Step 2: Create parser
parser = PydanticOutputParser(pydantic_object=Facts)

# Step 3: Create model
model = ChatOpenAI()

# Step 4: Prompt template
template = PromptTemplate(
    template="Give me 3 facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "football"})

print(result)