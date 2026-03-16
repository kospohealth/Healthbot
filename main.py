import os
import fitz  # PyMuPDF 라이브러리
from fastapi import FastAPI, Request
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

app = FastAPI()

# 1. 재료 준비 (Gemini & ChromaDB 설정)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 문장을 숫자로 바꿔주는 엔진 (기본값 사용, 성능 더 높이려면 Google 임베딩 모델 사용 권장)
emb_fn = embedding_functions.DefaultEmbeddingFunction()
# 데이터를 저장할 폴더 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_library", embedding_function=emb_fn)

# 2. PDF 파일에서 텍스트 추출하는 함수
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# 3. 'documents' 폴더의 PDF 읽어서 ChromaDB에 넣기 (서버 시작 시 1회 실행)
DOC_DIR = "documents" # PDF 파일들이 들어있는 폴더 이름

@app.on_event("startup")
def load_docs_to_db():
    # documents 폴더가 없으면 에러 나므로 예외 처리
    if not os.path.exists(DOC_DIR):
        print(f"'{DOC_DIR}' 폴더가 없습니다. PDF를 로드하지 않습니다.")
        return

    # 이미 문서가 있으면 추가로 넣지 않음 (최초 1회만 실행하기 위해)
    if collection.count() > 0:
        print(f"이미 {collection.count()}개의 문서가 DB에 있습니다. 로드를 건너뜁니다.")
        return

    print("PDF 문서 로드 중...")
    pdf_files = [f for f in os.listdir(DOC_DIR) if f.endswith('.pdf')]
    
    loaded_docs = []
    loaded_ids = []

    for idx, filename in enumerate(pdf_files):
        pdf_path = os.path.join(DOC_DIR, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            loaded_docs.append(pdf_text)
            loaded_ids.append(f"pdf_doc_{idx}")

    if loaded_docs:
        collection.add(
            documents=loaded_docs,
            ids=loaded_ids
        )
        print(f"{len(loaded_docs)}개의 PDF 문서를 DB에 로드했습니다!")
    else:
        print("로드할 PDF 파일이 없습니다.")

# 4. 질문을 받으면 문서 검색 후 답변 생성
async def get_answer_from_docs(user_query: str):
    # (1) 질문과 비슷한 문서 3개 찾기
    results = collection.query(query_texts=[user_query], n_results=3)
    retrieved_docs = results['documents'][0] if results['documents'] else ["관련 정보를 찾을 수 없습니다."]
    
    # 여러 문서를 하나의 텍스트로 합치기
    context = "\n\n---\n\n".join(retrieved_docs)
    
    # (2) Gemini에게 검색한 정보를 주고 답변 요청
    prompt = f"""당신은 회사의 친절한 AI 비서입니다. 
아래의 [참고 문서] 내용을 바탕으로 직원의 질문에 답하세요.
문서에 없는 내용은 지어내지 말고 "담당 부서에 문의해주세요"라고 답하세요.

[참고 문서]
{context}

직원 질문: {user_query}"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text

# 5. 카카오톡 웹훅 엔드포인트
@app.post("/webhook")
async def kakao_webhook(req: Request):
    data = await req.json()
    user_msg = data.get('userRequest', {}).get('utterance', '')

    answer = await get_answer_from_docs(user_msg)

    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
    }

@app.get("/")
def home():
    return {"status": "AI Server is running"}
