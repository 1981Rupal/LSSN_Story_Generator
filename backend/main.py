from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os

# Import our services
from lssn_service import LSSNService
from story_engine import StoryEngine

app = FastAPI(title="LSSN Story Generator API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
lssn_service = LSSNService()
story_engine = StoryEngine()

# Mount generated assets to serve images
os.makedirs("generated_assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="generated_assets"), name="assets")

class StoryRequest(BaseModel):
    prompt: str
    genre: str = "Fantasy"
    pages: int = 5

class Page(BaseModel):
    page_number: int
    text: str
    image_prompt: str
    image_url: Optional[str] = None

class StoryResponse(BaseModel):
    title: str
    genre: str
    pages: List[Page]

@app.post("/generate/story", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    story_data = story_engine.generate_story_structure(request.prompt, request.genre, request.pages)
    return story_data

@app.post("/generate/visualize")
async def generate_visuals(
    prompt: str = Form(...),
    character_label: Optional[str] = Form(None),
    subject_image: UploadFile = File(None)
):
    # If a subject image is provided, save it temporarily
    subject_image_path = None
    if subject_image:
        temp_path = f"temp_{subject_image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(subject_image.file, buffer)
        subject_image_path = temp_path

    # Generate Image using LSSN
    output_path = lssn_service.generate_image(prompt, subject_image_path, character_label)
    
    # Clean up temp file
    if subject_image_path and os.path.exists(subject_image_path):
        os.remove(subject_image_path)
        
    # Return URL (assuming localhost for now, functionality to be refined)
    filename = os.path.basename(output_path)
    return {"image_url": f"http://localhost:8000/assets/{filename}"}

class VideoRequest(BaseModel):
    story_data: dict
    image_map: dict

@app.post("/generate/video")
async def generate_video(request: VideoRequest):
    video_path = lssn_service.generate_video_slideshow(request.story_data, request.image_map)
    if not video_path:
        raise HTTPException(status_code=500, detail="Video generation failed")
    
    video_url = f"http://localhost:8000/assets/{os.path.basename(video_path)}"
    return {"video_url": video_url}

@app.get("/")
def read_root():
    return {"status": "LSSN Story Generator API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
