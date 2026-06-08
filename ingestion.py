from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read PDF
converter = DocumentConverter()

result = converter.convert("documents/Tutorial_Short.pdf")

text = result.document.export_to_markdown()

print("Total Characters:", len(text))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Show first chunk
print("\nFirst Chunk:\n")
print(chunks[0])
