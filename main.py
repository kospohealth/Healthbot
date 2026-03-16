from fastapi import FastAPI
import google.generativeai as genai
import os

app = FastAPI()

# Gemini API 설정
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/")
def root():
    response = model.generate_content("건강검진 전에 금식은 몇시간 해야해?")
    return {"answer": response.text} "text": answer
                    }
                }
            ]
        }
    }
