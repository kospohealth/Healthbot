import os
import fitz  # PyMuPDF
from fastapi import FastAPI, Request
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. 설정 (모델 ID를 단순하게 변경)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ⚠️ 에러 발생 지점: 모델 ID를 "gemini-1.5-flash"로 설정
# 만약 계속 에러가 나면 나중에 "models/gemini-1.5-flash"로 시도해야 할 수도 있습니다.
MODEL_ID = "gemini-1.5-flash" 

emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc: text += page.get_text()
        doc.close()
    except Exception as e: print(f"PDF Error: {e}")
    return text

@app.on_event("startup")
def load_docs():
    DOC_DIR = "." # 파일이 있는 현재 위치
    if collection.count() > 0: return
    
    pdf_files = [f for f in os.listdir(DOC_DIR) if f.endswith('.pdf')]
    for idx, filename in enumerate(pdf_files):
        content = extract_text_from_pdf(os.path.join(DOC_DIR, filename))
        if content:
            collection.add(documents=[content], ids=[f"id_{idx}"])
            print(f"Loaded: {filename}")

@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        user_msg = data.get('userRequest', {}).get('utterance', '')

        # 문서 검색
        results = collection.query(query_texts=[user_msg], n_results=2)
        context = "\n".join(results['documents'][0]) if results['documents'] else ""

        # 답변 생성
        prompt = f"아래 문서를 참고해 답변하세요.\n\n[문서]\n{context}\n\n질문: {user_msg}"
        
        # ✅ 모델 호출 방식을 가장 표준적인 방식으로 변경
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        answer = response.text

    except Exception as e:
        # 에러 내용을 카카오톡으로 직접 확인하기 위해 메시지에 포함
        answer = f"모델 호출 오류: {str(e)}"

    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer}}]}}

@app.get("/")
def health(): return {"status": "ok", "count": collection.count()}
