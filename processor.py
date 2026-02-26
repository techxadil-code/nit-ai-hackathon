import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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

class ChecklistItem(BaseModel):
    item_name: str = Field(description="The required item from the checklist schema.")
    status: str = Field(description="Must be exactly one of: 'PRESENT', 'MISSING', 'PARTIAL'.")
    details: str = Field(description="If PRESENT or PARTIAL, briefly describe what was found. If MISSING, leave empty or say 'Not detected'.")

class FinalOutput(BaseModel):
    summary: CaseSummary = Field(description="Output A: Structured Case Summary")
    classification: CrimeClassification = Field(description="Output B: Crime Classification")
    checklist: list[ChecklistItem] = Field(description="Output C: Missing Items Checklist mapped to the detected crime type schema.")

def process_chargesheet_text(text: str, api_key: str) -> dict:
    """
    Sends the raw text to Gemini and extracts Structured JSON outputs A, B, and C.
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is missing.")

    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Indian legal assistant. Review the following Hindi police chargesheet (Case Diary).
    Perform the following tasks and return the result strictly matching the provided JSON schema.
    
    # Task 1: Generate a Structured Case Summary (Output A)
    Extract the FIR number, police station, accused name(s), victim/complainant name(s), comprehensive incident facts, and legal sections applied.
    
    # Task 2: Crime Classification (Output B)
    Classify the crime into one of: 'Theft / Robbery', 'Assault / Hurt', 'Cyber Fraud', 'NDPS', or 'UNKNOWN' if no clear match.
    
    # Task 3: Missing Items Checklist (Output C)
    Based on your classification, check the presence of these specific required items for the crime type as per the schema:
    {json.dumps(CHECKLISTS, indent=2)}
    
    For each required item in the corresponding category, determine if it is:
    - PRESENT (evidence clearly found, provide brief detail)
    - MISSING (item not detected)
    - PARTIAL (referenced but incomplete, provide detail on what is missing)
    
    Ensure you return exactly the schema requested.
    
    --- RAW CHARGESHEET TEXT ---
    {text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinalOutput,
            temperature=0.1, # Low temperature for factual extraction
        ),
    )
    
    # Try parsing the JSON response
    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"Failed to parse Gemini response: {e}")
        return {}
