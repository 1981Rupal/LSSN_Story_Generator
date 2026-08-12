# LSSN Story Generator

An AI-powered story generator with a FastAPI backend and a React (Vite) frontend. The backend uses a custom LSSN model plus Hugging Face `diffusers`/`transformers` to turn a story prompt into generated narrative and visual assets; the frontend provides the UI for creating and viewing stories.

## Structure

- `backend/` — FastAPI service (`main.py`), story engine, and the LSSN model/training code.
- `frontend/` — React + Vite + Tailwind UI.

## Getting started

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

`start_app.bat` launches both on Windows.
