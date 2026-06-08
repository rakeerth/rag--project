from pypdf import PdfReader, PdfWriter

reader = PdfReader("documents/Tutorial_EDIT.pdf")
writer = PdfWriter()

# Keep first 25 pages
for page_num in range(25):
    writer.add_page(reader.pages[page_num])

with open("documents/Tutorial_Short.pdf", "wb") as output_file:
    writer.write(output_file)

print("Created Tutorial_Short.pdf")
