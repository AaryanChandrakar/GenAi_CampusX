import os

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = OpenAI(model=model_name)

result = llm.invoke("What is the capital of France?")

print(result)
