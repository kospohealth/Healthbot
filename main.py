from fastapi import FastAPI
import os
import google.genai as genai

app = FastAPI()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/")
def root():
    response = model.generate_content("건강검진 전 금식은 몇 시간 해야 하나요?")
    return {"answer": response.text}