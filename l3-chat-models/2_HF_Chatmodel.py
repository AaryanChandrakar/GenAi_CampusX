import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")
hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
hf_provider = os.getenv("HF_PROVIDER", "together")

llm = HuggingFaceEndpoint(
    repo_id=hf_model,
    provider=hf_provider,
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a caption on cricket.")

print(result.content)
