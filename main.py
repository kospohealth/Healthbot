import os
import fitz
import google.generativeai as genai
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Gemini 설정 (성공하셨던 그 이름 그대로 사용!)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# 404를 해결한 가장 확실한 경로명입니다.
MODEL_NAME = 'models/gemini-1.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

# 2. 벡터 DB 설정
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 3. 문서 로드 (시작 시 한 번만 실행)
@app.on_event("startup")
def load_docs():
    if collection.count() > 0:
        return
    for f in os.listdir("."):
        if f.endswith(".pdf"):
            doc = fitz.open(f)
            text = "".join([page.get_text() for page in doc])
            collection.add(documents=[text], ids=[f])
            doc.close()

# 4. 핵심 웹훅 (속도 최적화 버전)
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        query = data.get('userRequest', {}).get('utterance', '')
        
        # [속도 향상 1] 검색 결과를 딱 1개만 가져와서 시간을 단축합니다.
        results = collection.query(query_texts=[query], n_results=1)
        context = results['documents'][0][0] if results['documents'] and results['documents'][0] else ""
        
        # [속도 향상 2] Gemini에게 '짧게' 대답하라고 강하게 요청합니다.
        prompt = f"문서내용: {context}\n질문: {query}\n위 내용을 바탕으로 '한 문장'으로만 짧게 답하세요."
        
        # [속도 향상 3] 생성 토큰 수를 제한하여 5초 이내에 끊습니다.
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 100, # 답변 길이를 제한하여 생성 시간을 줄임
                "temperature": 0.1
            }
        )
        answer = response.text.strip()
        
    except Exception as e:
        # 에러 발생 시에도 규격에 맞는 응답을 보내야 챗봇이 멍때리지 않습니다.
        answer = "잠시 후 다시 시도해 주세요."

    # 카카오톡 최종 응답 규격
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
    }

@app.get("/")
def health():
    return {"status": "ok", "count": collection.count()}
