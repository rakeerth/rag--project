from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

text = "Python is a programming language"

embedding = embedding_model.embed_query(text)

print("Embedding Dimension:", len(embedding))
print("First 10 Values:")
print(embedding[:10])
