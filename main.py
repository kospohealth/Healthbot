import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Gemini 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# 2. 벡터 DB 설정
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="health_docs",
    embedding_function=emb_fn
)


def split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    긴 텍스트를 chunk_size 단위로 자르되,
    문맥 유지를 위해 overlap 만큼 겹치게 분할
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


# 3. 문서 로드
@app.on_event("startup")
def load_docs():
    print("🚀 [시스템] PDF 학습 프로세스 시작...")

    try:
        current_count = collection.count()
        if current_count > 0:
            print(f"✅ [시스템] 이미 {current_count}개의 데이터가 저장되어 있습니다.")
            print("ℹ️ [안내] 새 구조로 다시 학습하려면 chroma_db 폴더를 삭제 후 재배포하세요.")
            return
    except Exception as e:
        print(f"⚠️ [경고] 기존 DB 확인 중 오류: {e}")

    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ [경고] 루트 폴더에 PDF 파일이 하나도 없습니다!")
        return

    total_chunks = 0

    for f in pdf_files:
        try:
            print(f"📖 [학습 중] 파일 읽기: {f}")
            doc = fitz.open(f)

            full_text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    full_text += page_text + "\n"

            doc.close()

            chunks = split_text(full_text, chunk_size=500, overlap=100)

            if not chunks:
                print(f"   ㄴ ⚠️ 텍스트 추출 실패 또는 빈 문서: {f}")
                continue

            for idx, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    ids=[f"{f}_{idx}"],
                    metadatas=[{"source": f, "chunk_index": idx}]
                )

            total_chunks += len(chunks)
            print(f"   ㄴ ✅ {f} 학습 완료! (총 {len(chunks)}개 조각 저장)")

        except Exception as e:
            print(f"   ㄴ ❌ {f} 학습 실패: {e}")

    print(f"🎉 [완료] 총 {total_chunks}개의 지식 조각을 확보했습니다.")


# 4. 웹훅
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        query = data.get("userRequest", {}).get("utterance", "").strip()
        print(f"📩 [사용자 질문] {query}")

        if not query:
            answer = "질문 내용을 확인하지 못했습니다. 다시 입력해주세요."
        else:
            # 관련 문서 검색
            results = collection.query(
                query_texts=[query],
                n_results=3
            )

            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])

            if not documents or not documents[0]:
                answer = "관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해주세요."
            else:
                context_parts = []
                for i, doc_text in enumerate(documents[0]):
                    source_info = ""
                    if metadatas and metadatas[0] and i < len(metadatas[0]):
                        meta = metadatas[0][i]
                        source_info = f"(출처: {meta.get('source', '알 수 없음')})\n"
                    context_parts.append(source_info + doc_text)

                context = "\n\n".join(context_parts)

                print("📚 [검색된 문맥]")
                print(context[:1000])  # 로그 너무 길어지는 것 방지

                prompt = f"""
[참고 문서]
{context}

[사용자 질문]
{query}

[답변 규칙]
- 반드시 참고 문서를 기반으로 답변하세요.
- 핵심 정보(대상, 금액, 조건, 시기 등)를 중심으로 설명하세요.
- 답변은 자연스러운 한국어 문장으로 작성하세요.
- 문서 문장을 그대로 복사하지 말고 이해해서 설명하세요.
- 답변은 2~3문장 이내로 작성하세요.
- 문장이 중간에 끊기지 않도록 완결된 문장으로 작성하세요.
- 인사말은 작성하지 마세요.
- 질문을 반복하지 마세요.
- 참고 문서에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다"라고 답하세요.
- 금액, 대상, 횟수, 조건처럼 중요한 정보가 있으면 우선 포함하세요.

[답변]
"""

                response = model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 200,
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                )

                answer = response.text.strip() if response.text else "문서에서 답변을 생성하지 못했습니다."

        print(f"🤖 [챗봇 답변] {answer}")

    except Exception as e:
        print(f"❌ [에러 로그] {e}")
        answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }
