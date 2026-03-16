import os
import fitz
import google.generativeai as genai  # 라이브러리 호출 방식 변경
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash') # 가장 안정적인 호출

emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 2. PDF 로드 (현재 폴더 ".")
@app.on_event("startup")
def load_docs():
    if collection.count() > 0: return
    for f in os.listdir("."):
        if f.endswith(".pdf"):
            doc = fitz.open(f)
            text = "".join([page.get_text() for page in doc])
            collection.add(documents=[text], ids=[f])
            print(f"Loaded: {f}")

# 3. 질문 응답
@app.post("/webhook")
async def kakao_webhook(req: Request):
    data = await req.json()
    query = data.get('userRequest', {}).get('utterance', '')
    
    # 검색
    results = collection.query(query_texts=[query], n_results=1)
    context = results['documents'][0][0] if results['documents'] else ""
    
    # 답변 생성
    prompt = f"문서내용: {context}\n질문: {query}\n친절하게 답해줘."
    response = model.generate_content(prompt)
    
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response.text}}]}
    }
