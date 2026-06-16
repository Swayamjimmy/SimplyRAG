import fitz  # PyMuPDF
import base64
import os
from groq import Groq


def extract_images_from_pdf(pdf_path):
    """Extract embedded images from PDF pages as base64."""
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get all images embedded on this page
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            # Extract raw image bytes using the xref ID
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Encode as base64 for the multimodal API
            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            images.append({
                "page_num": page_num + 1,
                "image_index": img_index,
                "base64_data": base64_data
            })

    doc.close()
    return images

def extract_tables_from_pdf(pdf_path):
    """Extract table structures from PDF pages using PyMuPDF table detection."""
    tables = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use PyMuPDF built-in table detection (v1.23.0+)
        page_tables = page.find_tables()

        for table_index, table in enumerate(page_tables):
            # Convert to markdown for LLM-friendly token-efficient format
            table_text = table.to_markdown()

            tables.append({
                "page_num": page_num + 1,
                "table_index": table_index,
                "table_text": table_text
            })

    doc.close()
    return tables

def describe_image(base64_data, page_context, llm=None):
    """Generate a text description of an image using Groq Llama 4 Scout."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Send the base64 image to the multimodal model for description
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this image in detail for document search purposes. "
                                f"Page context: {page_context}. "
                                f"Include all visible data, labels, axes, legends, and key takeaways."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_data}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=1024
    )

    return response.choices[0].message.content

def extract_multimodal_content(pdf_path, llm=None):
    """Extract and describe all multimodal content from a PDF."""
    enriched_chunks = []

    # Generate text descriptions for each extracted image
    images = extract_images_from_pdf(pdf_path)
    for img in images:
        page_context = f"Page {img['page_num']} of {os.path.basename(pdf_path)}"
        description = describe_image(img["base64_data"], page_context, llm)

        enriched_chunks.append({
            "text": description,
            "metadata": {
                "source": pdf_path,
                "page_num": img["page_num"],
                "content_type": "image",
                "image_index": img["image_index"]
            }
        })

    # Convert extracted tables into searchable text chunks
    tables = extract_tables_from_pdf(pdf_path)
    for tbl in tables:
        enriched_chunks.append({
            "text": f"Table from page {tbl['page_num']}:\n{tbl['table_text']}",
            "metadata": {
                "source": pdf_path,
                "page_num": tbl["page_num"],
                "content_type": "table",
                "table_index": tbl["table_index"]
            }
        })

    return enriched_chunks