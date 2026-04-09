from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of India, and it is known for its rich history and vibrant culture.",
    "The city of Delhi is home to many historical landmarks, including the Red Fort and India Gate.",
    "Delhi is a bustling metropolis that offers a mix of modernity and tradition, making it a popular destination for tourists and locals alike."
]

results = embeddings.embed_documents(documents)


for result in results:
    print(str(result))
