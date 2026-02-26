import fitz

def extract_pdf(pdf_path, output_path):
    print(f"Extracting {pdf_path}...")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {output_path}")

try:
    extract_pdf("Case diary - 999-2020.pdf", "case_diary.txt")
    extract_pdf("ai_hackathon.pdf", "ai_hackathon.txt")
except Exception as e:
    print(f"Error: {e}")
