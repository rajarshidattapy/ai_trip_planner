from anthropic import Anthropic
from typing import Dict
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure Claude API
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

class TravelRequest(BaseModel):
    location: str
    budget: float
    days: int

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_prompt(location: str, budget: float, days: int) -> str:
    return f"""Create a detailed day-by-day travel itinerary for {location} for {days} days with a budget of ${budget}.
    For each day, please include:
    - Morning activities with estimated costs
    - Afternoon activities with estimated costs
    - Evening activities with estimated costs
    - Recommended restaurants for meals with price ranges
    - Local transportation tips
    Please ensure all suggestions fit within the total budget of ${budget}.
    Format the response in a clear, easy-to-read structure."""

async def get_itinerary(location: str, budget: float, days: int) -> str:
    try:
        message = await anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": generate_prompt(location, budget, days)
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-itinerary")
async def create_itinerary(request: TravelRequest):
    try:
        itinerary = await get_itinerary(
            request.location,
            request.budget,
            request.days
        )
        return {"itinerary": itinerary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)