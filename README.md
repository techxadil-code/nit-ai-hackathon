# Smart Chargesheet Review & Summarisation Assistant

This repository contains our submission for the AI/ML Hackathon: Smart Chargesheet Review & Summarisation Assistant. 
The application takes raw Hindi Police Chargesheets (Case Diaries) in PDF format, robustly extracts the text bridging OCR anomalies, and uses Gemini 1.5 Flash to automatically structure the case into predefined JSON schemas.

## Features Completed

| Stage | Feature | Status | Description |
|---|---|---|---|
| Stage 1 | Core Pipeline | ✅ Complete | Full end-to-end extraction and AI reasoning. |
| Stage 1 | Output A (Case Summary) | ✅ Complete | Extracts FIR, PS, Accused, Victim, Sections, and Facts. |
| Stage 1 | Output B (Crime Classification) | ✅ Complete | Classifies into 4 types or UNKNOWN w/ reasoning. |
| Stage 1 | Output C (Missing Items Checklist) | ✅ Complete | Dynamically maps `checklists.json` to extracted facts. |
| Stage 1 | Technical Requirements | ✅ Complete | Programmatic API via GenAI SDK, Structured JSON schemas, automated PDF cleaning (`PyMuPDF`). |

## Setup Instructions

**Prerequisites:** Python 3.9+

1. **Clone the repository** (if you haven't already).
2. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure the Environment:**
   Copy `.env.example` to `.env` and insert your API key:
   ```bash
   GEMINI_API_KEY=your_google_api_key_here
   ```
   *Alternatively, provide the key dynamically in the Streamlit Sidebar.*

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Repository Structure
- `app.py`: Streamlit User Interface.
- `extractor.py`: PDF OCR and text parsing utility (handles `fitz`).
- `processor.py`: LLM reasoning wrapper ensuring deterministic structured output handling.
- `checklists.json`: Authoritative config file representing the Ground Truth definitions.
- `requirements.txt`: Python package list.

## 3-Minute Demo Instructions
1. Run `streamlit run app.py`.
2. Browse files and upload `Case diary - 999-2020.pdf`.
3. Give it 5-10 seconds to intelligently unpack the Hindi Unicode structures using Gemini 1.5.
4. Review Output A, B, C side-by-side mapping correctly to the "Theft / Robbery" category explicitly mapping missing files (like seizure memos, chain of custody).
