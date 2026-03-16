from fastapi import FastAPI, Request
from google import genai
import os
import asyncio

app = FastAPI()

# ✅ 최신 SDK 스타일 (클라이언트 초기화)
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-1.5-flash"

company_docs = [
    "건강검진 전 금식은 일반적으로 8시간 이상 권장됩니다.",
    "혈압 측정 시 안정 상태에서 측정해야 합니다.",
    "소변 검사는 아침 첫 소변을 권장합니다."
]

def search_docs(query: str):
    # 간단한 키워드 포함 여부 검사 (실무에선 Vector DB 권장)
    relevant_docs = [doc for doc in company_docs if any(word in doc for word in query.split())]
    return relevant_docs if relevant_docs else company_docs

async def ask_gemini(question: str, context_docs: list):
    """비동기로 Gemini API 호출"""
    context_str = "\n".join(context_docs)
    prompt = f"당신은 회사의 건강검진 안내원입니다. 아래 자료를 참고하여 답변하세요.\n\n[참고 자료]\n{context_str}\n\n질문: {question}"
    
    # run_in_executor를 사용하거나 SDK의 async 기능을 활용
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: client.models.generate_content(model=MODEL_ID, contents=prompt)
    )
    return response.text

@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        user_msg = data.get('userRequest', {}).get('utterance', '')

        # 1️⃣ RAG 검색
        docs = search_docs(user_msg)

        # 2️⃣ Gemini 호출 (await 사용)
        answer = await ask_gemini(user_msg, docs)

        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer}}]
            }
        }
    except Exception as e:
        # 에러 발생 시 카카오톡이 요구하는 응답 형식 유지
        return {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": f"오류가 발생했습니다: {str(e)}"}}]}
        }
