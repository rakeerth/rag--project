from pypdf import PdfReader

reader = PdfReader("documents/Tutorial_EDIT.pdf")
print("Pages:", len(reader.pages))
