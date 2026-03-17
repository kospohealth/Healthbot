import google.generativeai as genai
import os

# 1. API 키 설정 (본인의 키를 직접 넣거나 환경변수 사용)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 사용 가능한 모델 목록 출력
print("--- 사용 가능한 모델 목록 ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Name: {m.name}") # 여기서 나오는 이름이 '진짜'입니다.
import os
import asyncio
import httpx
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, Request, BackgroundTasks
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Gemini 설정 (유료 티어용 2.0-flash-001 또는 최신 모델 권장)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

# 2. 벡터 DB 및 임베딩 설정
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="health_docs",
    embedding_function=emb_fn
)

# --- 유틸리티 함수 (기존 로직 유지) ---
def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    text = text.strip()
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += (chunk_size - overlap)
    return chunks

def clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def build_context(results) -> str:
    documents = results.get("documents", [[]])
    if not documents or not documents[0]: return ""
    seen = set()
    context_parts = []
    for doc_text in documents[0]:
        cleaned = clean_text(doc_text)
        if len(cleaned) < 20 or cleaned[:100] in seen: continue
        seen.add(cleaned[:100])
        context_parts.append(cleaned)
    return "\n\n".join(context_parts)

# 3. 문서 로드 (서버 시작 시 실행)
@app.on_event("startup")
def load_docs():
    print("🚀 [시스템] PDF 학습 프로세스 시작...")
    if collection.count() > 0:
        print(f"✅ [시스템] 기존 {collection.count()}개 데이터 사용")
        return

    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    for f in pdf_files:
        try:
            doc = fitz.open(f)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
            chunks = split_text(full_text)
            for idx, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    ids=[f"{f}_{idx}"],
                    metadatas=[{"source": f, "idx": idx}]
                )
            print(f"✅ {f} 학습 완료")
        except Exception as e:
            print(f"❌ {f} 실패: {e}")

# 4. 백그라운드 답변 생성 및 콜백 함수
async def process_and_callback(query: str, callback_url: str):
    try:
        # 지식 검색 (표 내용을 잘 읽기 위해 3개 추출)
        results = collection.query(query_texts=[query], n_results=3)
        context = build_context(results)

        if not context:
            answer = "관련 문서를 찾지 못했습니다. 질문을 구체적으로 입력해주세요."
        else:
# 수정된 프롬프트 예시
prompt = f"""
[참고 문서]
{context}

[사용자 질문]
{query}

[답변 가이드라인]
1. 당신은 'KOSPO 건강관리실'의 친절한 전문 상담원입니다. 
2. 참고 문서의 내용을 '핵심 근거'로 삼되, 사용자의 질문 의도를 파악하여 따뜻하고 능숙하게 답변하세요.
3. 문서의 딱딱한 말투를 지양하고, 동료에게 설명하듯 부드러운 구어체(~하세요, ~입니다 등)를 사용하세요.
4. 문서에 명시된 금액이나 날짜 등 '팩트'는 절대 왜곡하지 마세요.
5. 만약 문서 내용이 부족하다면, 관련 부서 연락처를 안내하거나 "일반적으로는 ~하지만 정확한 사항은 확인이 필요하다"는 식의 유연한 조언을 덧붙이세요.
6. 이모지를 적절히 섞어서 가독성을 높여주세요.

[답변]
"""
            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": 500, # 콜백이므로 넉넉하게
                    "temperature": 0.3
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            answer = response.text.strip()

        # 카카오톡 콜백 서버로 전송
        callback_payload = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer}}]
            }
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json=callback_payload, timeout=10.0)
            print(f"✅ [콜백 성공] 질의: {query[:10]}...")

    except Exception as e:
        print(f"❌ [콜백 에러] {e}")

# 5. 메인 웹훅 엔드포인트
@app.post("/webhook")
async def kakao_webhook(req: Request, background_tasks: BackgroundTasks):
    try:
        data = await req.json()
        query = data.get("userRequest", {}).get("utterance", "").strip()
        callback_url = data.get("userRequest", {}).get("callbackUrl")

        if callback_url:
            # 백그라운드에서 답변 생성 시작 (5초 타임아웃 우회)
            background_tasks.add_task(process_and_callback, query, callback_url)
            
            # 카톡 서버에는 즉시 "수락" 응답을 보냄
            return {
                "version": "2.0",
                "useCallback": True,
                "data": {"text": "건강관리실 봇이 답변을 준비 중입니다... (최대 1분 소요)"}
            }
        else:
            return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "콜백 설정이 필요합니다."}}]}}

    except Exception as e:
        print(f"❌ [웹훅 에러] {e}")
        return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "서버 오류 발생"}}]}}

@app.get("/")
def health():
    return {"status": "running", "db_count": collection.count()}
