from pathlib import Path
from pypdf import PdfReader
from mistralai.client import MistralClient
import numpy as np
import faiss
from mistralai.models.chat_completion import ChatMessage

pdf_files = Path("data").glob("*.pdf")
text = ""

for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
for page in reader.pages:
    text += page.extract_text() + "\n\n"

chunk_size = 500
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

client = MistralClient(api_key="MISTRAL_CHAT_BOT_KEY")


def embed(input: str):
    return client.embeddings("mistral-embed", input=input).data[0].embedding


embeddings = np.array([embed(chunk) for chunk in chunks])
dimension = embeddings.shape[1]

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

question = "What is the latest project Sofija worked on?"
question_embeddings = np.array([embed(question)])

D, I = index.search(question_embeddings, k=2)  # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-medium"):
    messages = [ChatMessage(role="user", content=user_message)]
    chat_response = client.chat(model=model, messages=messages)
    return chat_response.choices[0].message.content

print(run_mistral(prompt))