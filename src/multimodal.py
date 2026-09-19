<<<<<<< HEAD
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
=======
# src/multimodal.py

import base64
import os

import fitz  # PyMuPDF

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


# ============================================================
# Gemini Configuration
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your .env file or deployment secrets."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# Image Extraction
# ============================================================

def extract_images_from_pdf(
    pdf_path
):
    """
    Extract embedded images from PDF pages
    as base64 strings.
    """

    images = []

    doc = fitz.open(
        pdf_path
    )

    try:

        for page_num in range(
            len(doc)
        ):

            page = doc[
                page_num
            ]

            image_list = page.get_images(
                full=True
            )

            for img_index, img_info in enumerate(
                image_list
            ):

                # XRef ID of embedded image.
                xref = img_info[0]

                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image[
                    "image"
                ]

                image_ext = base_image.get(
                    "ext",
                    "png"
                )

                mime_type = (
                    f"image/{image_ext}"
                )

                # Convert image bytes
                # into base64 for compatibility
                # with the existing pipeline.

                base64_data = base64.b64encode(
                    image_bytes
                ).decode(
                    "utf-8"
                )

                images.append(
                    {
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "base64_data": base64_data,
                        "mime_type": mime_type,
                    }
                )

    finally:
        doc.close()

    return images


# ============================================================
# Table Extraction
# ============================================================

def extract_tables_from_pdf(
    pdf_path
):
    """
    Extract table structures from PDF pages
    using PyMuPDF's table detection.

    Tables are converted to Markdown so they can
    be stored and retrieved as searchable text.
    """

    tables = []

    doc = fitz.open(
        pdf_path
    )

    try:

        for page_num in range(
            len(doc)
        ):

            page = doc[
                page_num
            ]

            # PyMuPDF table detection.
            page_tables = page.find_tables()

            for table_index, table in enumerate(
                page_tables
            ):

                table_text = table.to_markdown()

                tables.append(
                    {
                        "page_num": page_num + 1,
                        "table_index": table_index,
                        "table_text": table_text,
                    }
                )

    finally:
        doc.close()

    return tables


# ============================================================
# Gemini Image Description
# ============================================================

def describe_image(
    base64_data,
    page_context,
    llm=None,
    mime_type="image/png"
):
    """
    Generate a searchable textual description
    of a PDF image using Gemini multimodal understanding.

    The llm argument is retained for compatibility
    with the existing code, but Gemini is handled
    directly through the Google GenAI SDK.
    """

    prompt = (
        "Describe this image in detail for document "
        "search purposes.\n\n"

        f"Page context: {page_context}\n\n"

        "Include:\n"
        "- all visible text\n"
        "- labels\n"
        "- axes\n"
        "- legends\n"
        "- numerical values\n"
        "- relationships between elements\n"
        "- important visual patterns\n"
        "- key takeaways\n\n"

        "Do not invent information that is not visible "
        "in the image."
    )

    # Decode the base64 representation.
    image_bytes = base64.b64decode(
        base64_data
    )

    # Gemini supports image bytes directly.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
    )

    return response.text


# ============================================================
# Multimodal PDF Processing
# ============================================================

def extract_multimodal_content(
    pdf_path,
    llm=None
):
    """
    Extract and describe all multimodal content
    from a PDF.

    Produces:
        - searchable image descriptions
        - searchable table Markdown
    """

    enriched_chunks = []

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    images = extract_images_from_pdf(
        pdf_path
    )

    for img in images:

        page_context = (
            f"Page {img['page_num']} "
            f"of {os.path.basename(pdf_path)}"
        )

        description = describe_image(
            img["base64_data"],
            page_context,
            llm=llm,
            mime_type=img.get(
                "mime_type",
                "image/png"
            ),
        )

        enriched_chunks.append(
            {
                "text": description,
                "metadata": {
                    "source": pdf_path,
                    "page_num": img["page_num"],
                    "content_type": "image",
                    "image_index": img["image_index"],
                },
            }
        )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    tables = extract_tables_from_pdf(
        pdf_path
    )

    for tbl in tables:

        enriched_chunks.append(
            {
                "text": (
                    f"Table from page "
                    f"{tbl['page_num']}:\n"
                    f"{tbl['table_text']}"
                ),
                "metadata": {
                    "source": pdf_path,
                    "page_num": tbl["page_num"],
                    "content_type": "table",
                    "table_index": tbl["table_index"],
                },
            }
        )
>>>>>>> hf-deploy

    return enriched_chunks