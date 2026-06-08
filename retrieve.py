from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

COLLECTION = "python_docs"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
client = MilvusClient(uri="http://localhost:19530")

query = "What is a Python list?"
q_vector = embedding_model.embed_query(query)   # no prefix this time

results = client.search(
    collection_name=COLLECTION,
    data=[q_vector],
    limit=5,
    output_fields=["text"],
)

for rank, hit in enumerate(results[0], 1):
    preview = hit["entity"]["text"].replace("\n", " ")[:200]
    print(f"\n#{rank}  score={hit['distance']:.3f}\n{preview}")
