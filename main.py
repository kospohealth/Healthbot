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
        
        # 1. 검색 결과 최적화 (가장 관련 있는 것만 빠르게)
        results = collection.query(query_texts=[query], n_results=1)
        context = results['documents'][0][0] if results['documents'] and results['documents'][0] else ""
        
        # 2. 프롬프트 최적화 (AI가 서론을 빼고 바로 결론을 말하게 하여 시간 단축)
        # "네, 안내해드릴게요" 같은 말을 빼는 것이 핵심입니다.
        prompt = f"문서내용: {context}\n질문: {query}\n위 내용을 바탕으로 서론 없이 결론만 바로 상세하게 답하세요. 반드시 완성된 문장으로 끝맺으세요."
        
        response = model.generate_content(
    prompt,
    generation_config={
        "max_output_tokens": 150, 
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 1
    }
)
        answer = response.text.strip()
        
    except Exception as e:
        print(f"❌ [에러 로그] {e}")
        answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    # 3. 답변이 너무 길면 카톡에서 거절할 수 있으므로 안전장치 추가
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer[:400]}}] # 카톡 한 글자 제한(보통 500자) 안전권
        }
    }
