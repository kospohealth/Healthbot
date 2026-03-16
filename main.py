import os
import fitz  # PyMuPDF
from fastapi import FastAPI, Request
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()

# 1. 설정 (Gemini API & 벡터 DB)
# 모델명을 "gemini-1.5-flash"로 정확히 지정합니다.
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-1.5-flash" 

emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 2. PDF에서 글자 추출 함수
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"PDF 읽기 에러 ({pdf_path}): {e}")
    return text

# 3. 서버 시작 시 PDF 로드 (DOC_DIR을 현재 폴더인 "."으로 설정)
@app.on_event("startup")
def load_docs():
    # PDF 파일들이 main.py와 같은 위치에 있으므로 "." 사용
    DOC_DIR = "." 
    
    # 중복 로드 방지
    if collection.count() > 0:
        print(f"이미 {collection.count()}개의 문서가 DB에 있습니다.")
        return

    print("PDF 문서 분석 및 DB 등록 중...")
    pdf_files = [f for f in os.listdir(DOC_DIR) if f.endswith('.pdf')]
    
    for idx, filename in enumerate(pdf_files):
        path = os.path.join(DOC_DIR, filename)
        content = extract_text_from_pdf(path)
        if content:
            # 문서가 너무 길면 잘라서 넣는 것이 좋으나, 우선 통째로 등록
            collection.add(documents=[content], ids=[f"id_{idx}"])
            print(f"등록 완료: {filename}")

# 4. 카카오톡 질문 처리 (RAG 방식)
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        user_msg = data.get('userRequest', {}).get('utterance', '')

        # (1) 문서 검색 (질문과 가장 관련 있는 내용 2개 추출)
        results = collection.query(query_texts=[user_msg], n_results=2)
        context = "\n".join(results['documents'][0]) if results['documents'] else "관련 문서를 찾지 못했습니다."

        # (2) Gemini 답변 생성
        prompt = f"""당신은 회사의 건강검진 및 안전 보건 담당자입니다. 
아래 제공된 [참고 문서]의 내용만을 바탕으로 직원의 질문에 친절하게 답하세요.
만약 문서에 답변할 내용이 없다면 "죄송하지만 해당 내용은 공문서에서 찾을 수 없습니다. 담당 부서에 문의해 주세요."라고 답하세요.

[참고 문서]
{context}

직원 질문: {user_msg}"""

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
        answer = response.text

    except Exception as e:
        answer = f"서버 오류가 발생했습니다: {str(e)}"

    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
    }

@app.get("/")
def health():
    return {"status": "ok", "docs_count": collection.count()}
