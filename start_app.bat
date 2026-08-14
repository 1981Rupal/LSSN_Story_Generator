@echo off
echo Starting LSSN Story Generator...

:: Start Backend in a new window
:: Start Backend in a new window
echo Installing backend dependencies...
cd backend && pip install -r requirements.txt && start cmd /k "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
cd ..

:: Start Frontend in a new window
start cmd /k "cd frontend && npm run dev"

echo Application launching...
echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:5173
pause
