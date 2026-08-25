from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import io
import os
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "CarFinder Backend Running"}


@app.post("/detect")
async def detect_vehicle(file: UploadFile = File(...)):
    try:

        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes))

        prompt = """
You are an automotive expert.

Analyze this vehicle image.

Return ONLY valid JSON.

{
    "make":"",
    "model":"",
    "color":"",
    "vehicle_type":"",
    "estimated_year":"",
    "confidence":"",
    "description":""
}

Rules:
- No markdown
- No explanation
- JSON only
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                prompt,
                image
            ]
        )

        text = response.text.strip()

        # Remove markdown if Gemini accidentally returns it
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )