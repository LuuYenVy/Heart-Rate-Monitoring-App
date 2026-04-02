from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()

print("ENV:", os.getenv("OPENAI_API_KEY"))

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set timeout (quan trọng)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=10.0  # ⏱️ 10 giây timeout
)

class AIRequest(BaseModel):
    heartRate: list[int]
    question: str

@app.post("/analyze")
async def analyze(data: AIRequest):

    logging.info("➡️ Received request")
    logging.info(f"HeartRate: {data.heartRate}")
    logging.info(f"Question: {data.question}")

    prompt = f"""
Heart rate data:
{data.heartRate}

Question:
{data.question}

Provide general health advice in Vietnamese. Do not diagnose.
"""

    try:
        logging.info("🚀 Calling OpenAI API...")

        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant. Provide safe general advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        end_time = time.time()
        logging.info(f"✅ OpenAI response received in {end_time - start_time:.2f}s")

        result = response.choices[0].message.content

        return {"result": result}

    except Exception as e:
        logging.error("❌ Error occurred:")
        logging.error(str(e))

        return {
            "error": str(e)
        }