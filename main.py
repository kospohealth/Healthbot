from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook")
async def webhook(req: Request):
    body = await req.json()

    user_msg = body["userRequest"]["utterance"]

    answer = f"질문을 받았어요: {user_msg}"

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
