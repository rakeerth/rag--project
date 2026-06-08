from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType
import os
import glob

COLLECTION = "python_docs"
DIM = 384
DOCS_FOLDER = "documents"

# Docling configured lightweight (no OCR / no page rendering) to avoid memory issues
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = False
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
        )
    }
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# ------------------------------------------------------------
# 1. Load EVERY PDF in the documents folder, keeping page numbers
# ------------------------------------------------------------
all_chunks = []   # each item: {"text", "source", "page"}

pdf_paths = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
print("PDFs found:", [os.path.basename(p) for p in pdf_paths])

for pdf_path in pdf_paths:
    source_name = os.path.basename(pdf_path)
    doc = converter.convert(pdf_path).document

    # group this document's text by page number
    pages = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None)
        page_no = prov[0].page_no if prov else 1
        pages.setdefault(page_no, []).append(text)

    # chunk each page separately so every chunk keeps its page number
    page_chunk_count = 0
    for page_no in sorted(pages):
        page_text = "\n".join(pages[page_no])
        for chunk in splitter.split_text(page_text):
            all_chunks.append(
                {"text": chunk, "source": source_name, "page": page_no})
            page_chunk_count += 1

    print(f"  {source_name}: {len(pages)} pages, {page_chunk_count} chunks")

print("Total chunks:", len(all_chunks))

# ------------------------------------------------------------
# 2. Embed
# ------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
embeddings = embedding_model.embed_documents([c["text"] for c in all_chunks])
print("Embeddings created")

# ------------------------------------------------------------
# 3. Connect + fresh collection with metadata columns
# ------------------------------------------------------------
client = MilvusClient(uri="http://localhost:19530")

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("text", DataType.VARCHAR, max_length=2000)
schema.add_field("source", DataType.VARCHAR,
                 max_length=256)   # metadata: file name
# metadata: page number
schema.add_field("page", DataType.INT64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)

index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding",
                       index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection(COLLECTION, schema=schema, index_params=index_params)

# ------------------------------------------------------------
# 4. Insert text + metadata + vector
# ------------------------------------------------------------
rows = [
    {"text": all_chunks[i]["text"], "source": all_chunks[i]["source"],
     "page": all_chunks[i]["page"], "embedding": embeddings[i]}
    for i in range(len(all_chunks))
]
client.insert(collection_name=COLLECTION, data=rows)
client.flush(COLLECTION)

print("Stored in Milvus successfully!")
print("Row count:", client.get_collection_stats(COLLECTION)["row_count"])
