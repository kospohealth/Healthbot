from fastapi import FastAPI, Request
import os
import google.genai as genai
from chromadb import Client  # RAG용 간단 예제

# FastAPI 서버
app = FastAPI()

# Gemini API 키
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# RAG 벡터 DB 초기화 (샘플)
chroma_client = Client()
# company_docs 인덱싱 → RAG 검색용
# 실제 구현 시 company_docs의 PDF/TXT 업로드 후 벡터 변환 필요

def search_docs(query: str):
    """질문과 관련된 회사 자료 검색 (RAG 샘플)"""
    # 실제: chroma_client.query(...)
    # 예제에서는 고정 샘플 반환
    return ["건강검진 전 금식은 8시간 이상 권장"]

def ask_gemini(question: str, context_docs: list):
    """Gemini API 호출"""
    prompt = f"다음 자료만 참고해서 답변하세요:\n{context_docs}\n질문: {question}"
    response = model.generate_content(prompt)
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