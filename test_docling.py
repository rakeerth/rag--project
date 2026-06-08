from docling.document_converter import DocumentConverter

# PDF path
pdf_path = "documents/Tutorial_Short.pdf"

# Create converter
converter = DocumentConverter()

# Convert PDF
result = converter.convert(pdf_path)

# Export text
text = result.document.export_to_markdown()

# Print first 2000 characters
print(text[:2000])
