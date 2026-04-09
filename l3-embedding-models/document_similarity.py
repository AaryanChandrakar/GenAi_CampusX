from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)

documents = [
    "Delhi is the capital of India, and it is known for its rich history and vibrant culture.",
    "The city of Delhi is home to many historical landmarks, including the Red Fort and India Gate.",
    "Delhi is a bustling metropolis that offers a mix of modernity and tradition, making it a popular destination for tourists and locals alike."
]

query = "List of historical landmarks in Delhi."

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)     

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[0]

print("User Query: ", query)
print("Most Similar Document: ")
print(documents[index])
print("Similarity score is: ", score)



