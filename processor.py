import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extractor import chunk_text

# Load checklists.json once
try:
    with open("checklists.json", "r", encoding="utf-8") as f:
        CHECKLISTS = json.load(f)
except FileNotFoundError:
    CHECKLISTS = {}

# Define Pydantic Schema for Gemini Structured Output
class CaseSummary(BaseModel):
    fir_number: str = Field(description="The FIR number and date")
    police_station: str = Field(description="The name of the police station")
    accused_names: list[str] = Field(description="List of accused names. If unknown, list as 'Unknown'.")
    victim_names: list[str] = Field(description="List of victim/complainant names.")
    incident_facts: str = Field(description="Comprehensive summary of the incident facts.")
    legal_sections: list[str] = Field(description="List of legal sections applied (e.g., IPC 379).")

class CrimeClassification(BaseModel):
    crime_type: str = Field(description="Classify into one of: 'Theft / Robbery', 'Assault / Hurt', 'Cyber Fraud', 'NDPS'. If no confident match, use 'UNKNOWN'.")
    reason: str = Field(description="Reason for the classification based on sections or facts. If UNKNOWN, return 'No matching sections found'.")

# -----------------
# STAGE 2A: NER Model
# -----------------
class Entity(BaseModel):
    text: str = Field(description="The matched Hindi or English text representing the entity.")
    type: str = Field(description="Must be one of: PERSON, LEGAL_SECTION, DATE_TIME, LOCATION, DOCUMENT, AMOUNT")
    role: str = Field(description="Role of the entity, e.g., 'ACCUSED', 'FIR_DATE', 'STOLEN_PROPERTY_VALUE'")

class FinalOutput(BaseModel):
    summary: CaseSummary = Field(description="Output A: Structured Case Summary")
    classification: CrimeClassification = Field(description="Output B: Crime Classification")
    entities: list[Entity] = Field(description="Stage 2A Bonus: Extracted Named Entities with their roles.")


# -----------------
# STAGE 2B logic
# -----------------
def get_embeddings(texts: list[str], client: genai.Client) -> list[list[float]]:
    """Helper to fetch embeddings using text-embedding-004"""
    # Gemini API expects list of strings for batch embedding request.
    try:
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=texts,
        )
        # Handle single vs batch results
        if hasattr(response, 'embeddings'):
             return [emb.values for emb in response.embeddings]
        else: # Handle list backwards compat
             return [emb.values for emb in response]
    except Exception as e:
        print(f"Embedding API Error: {e}")
        return []

def process_chargesheet_text(text: str, api_key: str) -> dict:
    """
    Sends the raw text to Gemini and extracts Structured JSON outputs A, B, and C.
    Integrates Stage 2A (NER) into the prompt and calculates Stage 2B (Semantic Match) deterministically.
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is missing.")

    client = genai.Client(api_key=api_key)
    
    # --- PHASE 1: Structure Extraction & NER (Stage 1A, 1B, 2A) ---
    prompt = f"""
    You are an expert Indian legal assistant. Review the following Hindi police chargesheet (Case Diary).
    Perform the following tasks and return the result strictly matching the provided JSON schema.
    
    # Task 1: Generate a Structured Case Summary (Output A)
    Extract the FIR number, police station, accused name(s), victim/complainant name(s), comprehensive incident facts, and legal sections applied.
    
    # Task 2: Crime Classification (Output B)
    Classify the crime into one of: 'Theft / Robbery', 'Assault / Hurt', 'Cyber Fraud', 'NDPS', or 'UNKNOWN' if no clear match.
    
    # Task 3: Stage 2A - Named Entity Recognition
    Extract all explicitly identifiable entities with their proper type and context role.
    Ensure "type" is strictly one of: PERSON, LEGAL_SECTION, DATE_TIME, LOCATION, DOCUMENT, AMOUNT.
    
    Ensure you return exactly the schema requested.
    
    --- RAW CHARGESHEET TEXT ---
    {text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalOutput,
                temperature=0.1, 
            ),
        )
        result_dict = json.loads(response.text)
    except Exception as e:
        print(f"Failed to parse Gemini struct response: {e}")
        return {}

    # --- PHASE 2: Semantic Similarity Scoring (Stage 2B / Output C) ---
    crime_type = result_dict.get("classification", {}).get("crime_type")
    
    # Map the crime classification to our JSON schema keys
    schema_key_map = {
        "Theft / Robbery": "theft_robbery",
        "Assault / Hurt": "assault_hurt",
        "Cyber Fraud": "cyber_fraud",
        "NDPS": "ndps"
    }
    
    required_items = []
    if crime_type in schema_key_map:
        key = schema_key_map[crime_type]
        required_items = CHECKLISTS.get(key, {}).get("required_items", [])
    
    checklist_output = []
    
    if required_items:
        # Precompute chunks and embeddings
        chunks = chunk_text(text, chunk_size=300, overlap=50) 
        
        # Batch Embeddings to speed up API calls
        chunk_embs = get_embeddings(chunks, client)
        req_embs = get_embeddings(required_items, client)
        
        if chunk_embs and req_embs:
            chunk_embs_np = np.array(chunk_embs)
            req_embs_np = np.array(req_embs)
            
            # Compute similarity matrix: shape (num_requirements, num_chunks)
            similarity_matrix = cosine_similarity(req_embs_np, chunk_embs_np)
            
            for i, item in enumerate(required_items):
                scores = similarity_matrix[i]
                best_idx = np.argmax(scores)
                best_score = float(scores[best_idx])
                
                # Threshold for semantic match (e.g. 0.65 as suggested in hackathon doc)
                if best_score >= 0.65:
                    status = "PRESENT"
                    matched_text = chunks[best_idx]
                else:
                    status = "MISSING"
                    matched_text = ""
                    
                checklist_output.append({
                    "item": item,
                    "status": status,
                    "similarity_score": round(best_score, 2),
                    "matched_text": matched_text
                })
        else:
             print("Warning: Semantic embedding vectors came back empty; returning raw.")

    result_dict["checklist"] = checklist_output
    
    return result_dict
