import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 1. Gemini 설정 (가장 에러 없는 정식 모델명 사용)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-flash')

# 2. 벡터 DB 설정
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 3. 문서 로드 (로그 출력 강화 버전)
@app.on_event("startup")
def load_docs():
    print("🚀 [시스템] PDF 학습 프로세스 시작...")
    if collection.count() > 0:
        print(f"✅ [시스템] 이미 {collection.count()}개의 데이터가 저장되어 있습니다.")
        return
        
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ [경고] 루트 폴더에 PDF 파일이 하나도 없습니다!")
        return

    for f in pdf_files:
        try:
            print(f"📖 [학습 중] 파일 읽기: {f}")
            doc = fitz.open(f)
            text = "".join([page.get_text() for page in doc])
            collection.add(documents=[text], ids=[f])
            doc.close()
            print(f"   ㄴ ✅ {f} 학습 완료!")
        except Exception as e:
            print(f"   ㄴ ❌ {f} 학습 실패: {e}")

    print(f"🎉 [완료] 총 {collection.count()}개의 지식 조각을 확보했습니다.")

# 4. 웹훅 (에러 원인 추적 버전)
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        query = data.get("userRequest", {}).get("utterance", "").strip()

        # 사용자 입력이 비어 있을 때
        if not query:
            answer = "질문 내용을 확인하지 못했습니다. 다시 입력해주세요."
        else:
            # 1. 관련 문서 검색
            results = collection.query(query_texts=[query], n_results=3)
            context = "\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""

            # 2. 검색 결과가 없을 때
            if not context:
                answer = "관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해주세요."
            else:
                # 3. 프롬프트 생성
                prompt = f"""
[참고 문서]
{context}

[사용자 질문]
{query}

[답변 규칙]
- 반드시 참고 문서를 기반으로 답변하세요.
- 핵심 정보(대상, 금액, 조건 등)를 중심으로 설명하세요.
- 답변은 자연스러운 한국어 문장으로 작성하세요.
- 문서 문장을 그대로 복사하지 말고 이해해서 설명하세요.
- 답변은 2~3문장 이내로 작성하세요.
- 문장이 중간에 끊기지 않도록 완결된 문장으로 작성하세요.
- 인사말은 작성하지 마세요.
- 질문을 반복하지 마세요.
- 참고 문서에 없는 내용은 추측하지 말고, 문서에서 확인되지 않는다고 답하세요.

[답변]
"""

                # 4. Gemini 호출
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
