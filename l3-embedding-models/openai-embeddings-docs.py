from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "Delhi is the capital of India, and it is known for its rich history and vibrant culture.",
    "The city of Delhi is home to many historical landmarks, including the Red Fort and India Gate.",
    "Delhi is a bustling metropolis that offers a mix of modernity and tradition, making it a popular destination for tourists and locals alike."
]

result = embeddings.embed_documents(documents)

print(str(result))

