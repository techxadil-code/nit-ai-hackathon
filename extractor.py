import fitz  # PyMuPDF
import re

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from a PDF given as bytes.
    Cleans up common OCR and whitespace issues.
    """
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""
    
    # Basic Pre-processing:
    # 1. Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # 2. Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
