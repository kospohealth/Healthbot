import os
import httpx
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, Request, BackgroundTasks
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Gemini 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# 2. 벡터 DB 및 임베딩 설정
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="health_docs",
    embedding_function=emb_fn
)

# --- 유틸리티 함수 ---
def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += (chunk_size - overlap)
    return chunks


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def build_context(results) -> str:
    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return ""

    seen = set()
    context_parts = []

    for doc_text in documents[0]:
        cleaned = clean_text(doc_text)
        if len(cleaned) < 20:
            continue

        key = cleaned[:100]
        if key in seen:
            continue

        seen.add(key)
        context_parts.append(cleaned)

    return "\n\n".join(context_parts)


# 3. 문서 로드
@app.on_event("startup")
def load_docs():
    print("🚀 [시스템] PDF 학습 프로세스 시작...")

    if collection.count() > 0:
        print(f"✅ [시스템] 기존 {collection.count()}개 데이터 사용")
        return

    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ [시스템] 현재 폴더에 PDF 파일이 없습니다.")
        return

    for f in pdf_files:
        try:
            doc = fitz.open(f)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()

            chunks = split_text(full_text)

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                collection.add(
                    documents=[chunk],
                    ids=[f"{f}_{idx}"],
                    metadatas=[{"source": f, "idx": idx}]
                )

            print(f"✅ {f} 학습 완료")

        except Exception as e:
            print(f"❌ {f} 실패: {e}")


# 4. 백그라운드 답변 생성 및 콜백
async def process_and_callback(query: str, callback_url: str):
    try:
        results = collection.query(query_texts=[query], n_results=3)
        context = build_context(results)

        if not context:
            answer = "관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해주세요."
        else:
            prompt = f"""
[참고 문서]
{context}

[사용자 질문]
{query}

[답변 가이드라인]
1. 당신은 'KOSPO 건강관리실'의 친절한 전문 상담원입니다.
2. 참고 문서의 내용을 핵심 근거로 삼아 답변하세요.
3. 딱딱한 문장보다는 부드러운 안내문 스타일로 답변하세요.
4. 문서에 있는 금액, 날짜, 대상 기준은 절대 바꾸지 마세요.
5. 문서에 없는 내용은 추측하지 말고, 확인이 필요하다고 안내하세요.
6. 답변은 짧고 명확하게 2~4문장으로 작성하세요.
7. 필요할 때만 이모지를 1~2개 정도 사용하세요.

[답변]
"""

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.3,
                top_p=0.95,
                top_k=40,
            )

            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            answer = response.text.strip() if response.text else "답변을 생성하지 못했습니다."

        callback_payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer[:1000]  # 카카오 응답 길이 안전장치
                        }
                    }
                ]
            }
        }

        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json=callback_payload, timeout=10.0)

        print(f"✅ [콜백 성공] 질의: {query[:20]}")

    except Exception as e:
        print(f"❌ [콜백 에러] {e}")


# 5. 메인 웹훅
@app.post("/webhook")
async def kakao_webhook(req: Request, background_tasks: BackgroundTasks):
    try:
        data = await req.json()
        query = data.get("userRequest", {}).get("utterance", "").strip()
        callback_url = data.get("userRequest", {}).get("callbackUrl")

        if not query:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": "질문을 입력해주세요."}}
                    ]
                }
            }

        if callback_url:
            background_tasks.add_task(process_and_callback, query, callback_url)
            return {
                "version": "2.0",
                "useCallback": True,
                "data": {"text": "질문을 접수했어요. 곧 답변드릴게요."}
            }

        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "콜백 설정이 필요합니다."}}
                ]
            }
        }

    except Exception as e:
        print(f"❌ [웹훅 에러] {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "서버 오류가 발생했습니다."}}
                ]
            }
        }


@app.get("/")
def health():
    return {"status": "running", "db_count": collection.count()}
