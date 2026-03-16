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
        query = data.get('userRequest', {}).get('utterance', '')
        
        # 문서 검색 (가장 관련 있는 문단 1개 추출)
        results = collection.query(query_texts=[query], n_results=1)
        context = results['documents'][0][0] if results['documents'] and results['documents'][0] else ""
        
        # Gemini 답변 생성 (한 문장 제한으로 5초 타임아웃 방지)
        prompt = f"문서내용: {context}\n질문: {query}\n위 내용을 바탕으로 핵심 내용을 3문장 이내로 상세하고 친절하게 답하세요."
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 300,  # 500은 너무 길 수 있으니 300으로 타협합니다.
                "temperature": 0.1,         # 0.1로 낮추면 AI가 덜 고민하고 바로 답을 뱉습니다.
                "top_p": 0.8,
                "top_k": 40
            }
        )
        answer = response.text.strip()
        
    except Exception as e:
        # 에러가 발생하면 카톡 답변으로 원인을 직접 보냅니다.
        print(f"❌ [에러 로그] {e}")
        answer = f"오류 원인: {str(e)[:50]}... (로그 확인 필요)"
    
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
    }

@app.get("/")
def health():
    return {"status": "ok", "docs_count": collection.count()}
