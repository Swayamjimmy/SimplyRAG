import pymupdf as fitz  # PyMuPDF for reading PDF files
import os

def load_pdf(file_path):
    """Load a PDF file and extract text from each page."""
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "source": os.path.basename(file_path),
                "page": page_num + 1
            })
    doc.close()
    return pages

def chunk_text(pages, chunk_size=512, overlap=50):
    """Split page text into chunks with overlap and metadata."""
    chunks = []
    chunk_index = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1
            start += chunk_size - overlap
    return chunks

def ingest_pdfs(data_dir="data"):
    """Load all PDFs from a directory and return chunked text."""
    all_chunks = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            pages = load_pdf(file_path)
            chunks = chunk_text(pages)
            all_chunks.extend(chunks)
    print(f"Ingested {len(all_chunks)} chunks from {data_dir}")
    return all_chunks