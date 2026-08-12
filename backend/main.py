from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Optional
import shutil
import os

from lssn_service import LSSNService
from story_engine import StoryEngine

app = FastAPI(title="LSSN Story Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lssn_service = LSSNService()
story_engine = StoryEngine()

os.makedirs("generated_assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="generated_assets"), name="assets")

class StoryRequest(BaseModel):
    prompt: str
    genre: str = "Fantasy"
    # BUG FIX: no upper bound previously existed. Combined with the blocking
    # per-chapter fallback loop in StoryEngine, an unbounded `pages` value
    # let a single request queue up an arbitrary number of sequential LLM
    # calls against the (previously) blocked event loop.
    pages: int = Field(default=5, ge=1, le=20)

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
    # BUG FIX: generate_story_structure is synchronous, blocking (network) I/O.
    # Calling it directly from an async endpoint stalls the entire event loop
    # -- and therefore every other concurrent request -- for the duration of
    # the LLM call. run_in_threadpool offloads it to a worker thread.
    story_data = await run_in_threadpool(
        story_engine.generate_story_structure, request.prompt, request.genre, request.pages
    )
    return story_data

@app.post("/generate/visualize")
async def generate_visuals(
    prompt: str = Form(...),
    character_label: Optional[str] = Form(None),
    subject_image: UploadFile = File(None)
):
    subject_image_path = None
    if subject_image:
        temp_path = f"temp_{subject_image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(subject_image.file, buffer)
        subject_image_path = temp_path

    output_path = await run_in_threadpool(
        lssn_service.generate_image, prompt, subject_image_path, character_label
    )

    if subject_image_path and os.path.exists(subject_image_path):
        os.remove(subject_image_path)

    filename = os.path.basename(output_path)
    return {"image_url": f"http://localhost:8000/assets/{filename}"}

class VideoRequest(BaseModel):
    story_data: dict
    image_map: dict

@app.post("/generate/video")
async def generate_video(request: VideoRequest):
    video_path = await run_in_threadpool(
        lssn_service.generate_video_slideshow, request.story_data, request.image_map
    )
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
