from fastapi import FastAPI
import os
import google.genai as genai

app = FastAPI()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/")
def root():
    response = model.generate_content("건강검진 전 금식은 몇 시간 해야 하나요?")
    return {"answer": response.text}(prompt)
    return response.text

# 카카오톡 웹훅
@app.post("/webhook")
async def kakao_webhook(req: Request):
    data = await req.json()
    user_msg = data['userRequest']['utterance']

    # 1️⃣ RAG 검색
    docs = search_docs(user_msg)

    # 2️⃣ Gemini 호출
    answer = ask_gemini(user_msg, docs)

    # 3️⃣ 카카오톡용 JSON 반환
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": answer}}]}
    }

# 테스트용 루트
@app.get("/")
def root():
    return {"message": "Healthbot server is running"}