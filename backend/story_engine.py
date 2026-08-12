import random

class StoryEngine:
    def __init__(self):
        pass

    def _query_ollama(self, prompt: str, model: str = "llama3"):
        try:
            import requests
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            print(f"Ollama API Error: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Ollama Connection Error: {e}")
            return None

    def generate_story_structure(self, prompt: str, genre: str, num_pages: int):
        """
        Generates a structured story with scenes using local LLM (Ollama/Llama3).
        Falls back to templates if Ollama is unavailable.
        """
        import json
        
        # Try to generate using Ollama (Llama3)
        print(f"Attempting to generate story with Llama3 for prompt: {prompt}")
        
        # 1. Generate Title
        title_prompt = f"Generate a creative and short title for a {genre} story about: {prompt}. Return ONLY the title, no quotes or extra text."
        title = self._query_ollama(title_prompt)
        
        if not title:
            print("Llama3 unavailable or failed, falling back to templates.")
            return self._generate_fallback_template(prompt, genre, num_pages)
            
        print(f"Generated Title: {title}")

        story_data = {
            "title": title.replace('"', '').replace("Title:", "").strip(),
            "genre": genre,
            "pages": []
        }
        
        outline_prompt = (
            f"Create a {num_pages}-chapter outline for a {genre} story named '{title}' based on this premise: '{prompt}'. "
            f"The story must have exactly {num_pages} distinct scenes. "
            f"Return the response as a valid JSON object with a key 'chapters' which is a list of objects, "
            f"each containing 'chapter_number', 'plot_summary' (story text for that scene, approx 3-4 sentences), and 'visual_description' (a prompt for an image generator). "
            f"Do NOT include any markdown formatting like ```json. Just the raw JSON string."
        )
        
        outline_response = self._query_ollama(outline_prompt)
        
        if outline_response:
             # Robust cleanup of markdown formatting
            clean_json = outline_response
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            clean_json = clean_json.strip()
            
            try:
                outline_data = json.loads(clean_json)
                
                # Verify we got what we asked for
                if "chapters" in outline_data and isinstance(outline_data["chapters"], list):
                    for i, chapter in enumerate(outline_data["chapters"][:num_pages]):
                         page_content = {
                            "page_number": i + 1,
                            "text": chapter.get("plot_summary", "Story content missing..."),
                            "image_prompt": chapter.get("visual_description", f"Scene from {title}")
                        }
                         story_data["pages"].append(page_content)
                else:
                    raise ValueError("Invalid JSON structure")
                    
                # If we have less chapters than requested, pad it
                while len(story_data["pages"]) < num_pages:
                    idx = len(story_data["pages"]) + 1
                    story_data["pages"].append({
                        "page_number": idx,
                        "text": f"Chapter {idx} continues the story of {title}...",
                        "image_prompt": f"Scene from {title}, chapter {idx}"
                    })
                return story_data
                
            except Exception as e:
                print(f"Failed to parse LLM JSON: {e}. Raw response: {outline_response[:100]}...")
                # Fallback to per-chapter generation if JSON fails
        
        # Fallback loop integration (simpler but slower, or if JSON failed)
        current_context = f"Title: {title}\nPremise: {prompt}\nGenre: {genre}"
        
        for i in range(1, num_pages + 1):
            chapter_prompt = (
                f"Write Chapter {i} of {num_pages} for the story. \n"
                f"Context: {current_context} \n"
                f"Content: Write 3-4 sentences of the story for this chapter. \n"
                f"End with a separate line starting with 'VISUAL:' describing the scene for an image generator."
            )
            
            response = self._query_ollama(chapter_prompt)
            
            if response:
                parts = response.split("VISUAL:")
                text_part = parts[0].strip()
                visual_part = parts[1].strip() if len(parts) > 1 else f"Illustration of {text_part[:50]}..."
                
                # Update context slightly (naive)
                current_context += f"\nChapter {i}: {text_part}"
                
                story_data["pages"].append({
                    "page_number": i,
                    "text": text_part,
                    "image_prompt": visual_part
                })
            else:
                 story_data["pages"].append({
                    "page_number": i,
                    "text": "Chapter generation failed. Please try again.",
                    "image_prompt": "Error generating visual description."
                })

        return story_data

    def _generate_fallback_template(self, prompt: str, genre: str, num_pages: int):
        # ... (Previous template logic goes here as fallback) ...
        # (For brevity in this diff, reusing the logic from previous step but wrapped)
        
        templates = {
            "Fantasy": ["The ancient artifact began to hum...", "In the shadow of the Spire..."],
            "Sci-Fi": ["The ship's AI detected...", "On the neon-lit streets..."],
            "Mystery": ["The clue pointed to...", "A shadow moved..."],
            "Romance": ["They met by chance...", "A secret look..."],
            "Horror": ["Something creaked...", "Darkness fell..."]
        }
        # ... (rest of simple logic) ...
        
        title = f"The {genre} of {prompt.split(' ')[0]}" if prompt else "A Story"
        story_data = { "title": title, "genre": genre, "pages": [] }
        
        # Simple fill
        beats = templates.get(genre, templates["Mystery"])
        for i in range(num_pages):
             story_data["pages"].append({
                "page_number": i + 1,
                "text": f"Chapter {i+1}: {beats[i % len(beats)]} ({prompt})", # Simple fallback
                "image_prompt": f"{genre} scene page {i+1}"
             })
             
        return story_data
