import os
import fitz
import google.generativeai as genai
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. 설정 (모델 이름을 'models/' 포함해서 명확히 작성)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# ⚠️ 404 에러 방지를 위해 'models/'를 명시적으로 붙임
MODEL_NAME = 'models/gemini-1.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 2. PDF 로드 (현재 폴더 ".")
@app.on_event("startup")
def load_docs():
    try:
        if collection.count() > 0: return
        print("PDF 로드 시작...")
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        for f in pdf_files:
            doc = fitz.open(f)
            text = "".join([page.get_text() for page in doc])
            collection.add(documents=[text], ids=[f])
            print(f"성공적으로 로드됨: {f}")
    except Exception as e:
        print(f"로드 중 에러 발생: {e}")

# 3. 질문 응답
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        query = data.get('userRequest', {}).get('utterance', '')
        
        # 검색
        results = collection.query(query_texts=[query], n_results=1)
        context = results['documents'][0][0] if results['documents'] else ""
        
        # 답변 생성
        prompt = f"문서내용: {context}\n질문: {query}\n위 내용을 바탕으로 친절하게 답변해줘."
        response = model.generate_content(prompt)
        answer = response.text
        
    except Exception as e:
        # 에러가 나면 카톡으로 원인을 알려줌
        answer = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
    
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": answer}}]}
    }

@app.get("/")
def home(): return {"status": "ok", "docs": collection.count()}
