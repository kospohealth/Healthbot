import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, Request
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 1. 환경 설정 로드
load_dotenv()
app = FastAPI()

# 2. Gemini 설정 (가장 안정적인 google-generativeai 방식)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# 'models/'를 빼고 이름만 적어보세요
MODEL_NAME = 'gemini-1.5-flash'
model = genai.GenerativeModel('models/gemini-1.5-flash')

# 3. 벡터 DB 설정 (ChromaDB)
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="health_docs", embedding_function=emb_fn)

# 4. 서버 시작 시 PDF 문서 학습 (현재 폴더의 PDF 자동 탐색)
@app.on_event("startup")
def load_docs():
    try:
        # 이미 학습된 문서가 있다면 스킵
        if collection.count() > 0:
            print(f"이미 {collection.count()}개의 지식이 저장되어 있습니다.")
            return

        print("PDF 문서 학습을 시작합니다...")
        # 현재 폴더(.)에서 PDF 파일 목록 추출
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        
        if not pdf_files:
            print("학습할 PDF 파일이 없습니다.")
            return

        for f in pdf_files:
            doc = fitz.open(f)
            text = "".join([page.get_text() for page in doc])
            # 문서 내용을 DB에 저장
            collection.add(documents=[text], ids=[f])
            print(f"학습 완료: {f}")
            doc.close()
            
    except Exception as e:
        print(f"학습 중 에러 발생: {e}")

# 5. 카카오톡 웹훅 (질문 응답 처리)
@app.post("/webhook")
async def kakao_webhook(req: Request):
    try:
        data = await req.json()
        query = data.get('userRequest', {}).get('utterance', '')
        
        # (1) 문서 검색: 가장 관련 있는 내용 1개만 추출 (속도 향상)
        results = collection.query(query_texts=[query], n_results=1)
        context = results['documents'][0][0] if results['documents'] else "참고할 내용이 없습니다."
        
        # (2) Gemini 답변 생성: 5초 타임아웃 방지를 위해 '한 문장'으로 제한
        prompt = f"문서내용: {context}\n질문: {query}\n위 내용을 바탕으로 한 문장으로 친절하게 답하세요. 문서에 없는 내용은 모른다고 하세요."
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,  # 답변 길이를 제한하여 응답 속도 최적화
                temperature=0.1
            )
        )
        answer = response.text.strip()
        
    except Exception as e:
        print(f"에러 발생: {e}")
        answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    
    # (3) 카카오톡 규격에 맞춘 최종 응답 데이터
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

# 서버 상태 체크용 홈 화면
@app.get("/")
def home():
    return {"status": "ok", "docs_learned": collection.count()}
