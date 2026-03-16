from fastapi import FastAPI, Request
from google import genai
import os

app = FastAPI()

# ✅ Gemini 최신 SDK: 모델 생성 시 API 키 전달
model = genai.Model("gemini-1.5-flash", api_key=os.environ.get("GEMINI_API_KEY"))

# --- 회사 자료 기반 RAG 샘플 ---
# 실제 구현 시 chromadb + PDF/TXT 인덱싱 필요
company_docs = [
    "건강검진 전 금식은 일반적으로 8시간 이상 권장됩니다.",
    "혈압 측정 시 안정 상태에서 측정해야 합니다.",
    "소변 검사는 아침 첫 소변을 권장합니다."
]

def search_docs(query: str):
    """간단한 RAG 샘플: 질문과 관련된 문서 필터링"""
    return [doc for doc in company_docs if any(word in query for word in doc.split())] or company_docs

def ask_gemini(question: str, context_docs: list):
    """Gemini API 호출"""
    prompt = f"다음 회사 자료만 참고해서 답변하세요:\n{context_docs}\n질문: {question}"
    response = model.generate_text(prompt)
    return response.text

# --- 카카오톡 웹훅 엔드포인트 ---
@app.post("/webhook")
async def kakao_webhook(req: Request):
    data = await req.json()
    user_msg = data['userRequest']['utterance']

    # 1️⃣ RAG 검색
    docs = search_docs(user_msg)

    # 2️⃣ Gemini 호출
    answer = ask_gemini(user_msg, docs)

    # 3️⃣ 카카오톡 JSON 반환
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": answer}}
            ]
        }
    }

# --- 테스트용 루트 ---
@app.get("/")
def root():
    return {"message": "Healthbot server is running"}