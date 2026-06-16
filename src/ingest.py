import pymupdf as fitz  # PyMuPDF for reading PDF files
import os
from src.multimodal import extract_multimodal_content

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

def ingest_pdf(path="data"):
    """Load PDFs from either a directory or a single PDF file."""

    all_chunks = []

    # Single uploaded PDF
    if os.path.isfile(path):

        pages = load_pdf(path)

        text_chunks = chunk_text(pages)

        all_chunks.extend(text_chunks)

        multimodal_chunks = extract_multimodal_content(path)

        all_chunks.extend(multimodal_chunks)

        print(
            f"Ingested {len(all_chunks)} chunks from {path}"
        )

        return all_chunks

    # Directory of PDFs
    for filename in os.listdir(path):

        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(
            path,
            filename
        )

        pages = load_pdf(file_path)

        text_chunks = chunk_text(pages)

        all_chunks.extend(text_chunks)

        multimodal_chunks = extract_multimodal_content(
            file_path
        )

        all_chunks.extend(multimodal_chunks)

    print(
        f"Ingested {len(all_chunks)} chunks from {path}"
    )

    return all_chunks