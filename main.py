import os
import dotenv
import uvicorn

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from model import MachineTranslation

# Load environment variables from .env file
dotenv.load_dotenv()

app = FastAPI(
    title="Machine Translation",
    description="facebook/mbart-large-50-many-to-many-mmt",
    version="1.0.1"
)

model = MachineTranslation().load_model()


class RequestModel(BaseModel):
    text: str
    src_lang: str = "en"
    target_lang: str


class ResponseModel(BaseModel):
    source: str
    target: str


@app.post("/translate", response_model=ResponseModel)
def translate(payload: RequestModel):
    # Ensure text is provided
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Detect language
    try:
        target = model.translate(
            payload.text, payload.src_lang, payload.target_lang)
        return ResponseModel(source=payload.text, target=target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", '8000'))

    print("ReDoc OpenAPI: http://0.0.0.0:{}/redoc".format(port))
    print("Swagger OpenAPI: http://0.0.0.0:{}/docs".format(port))

    uvicorn.run(app, host="0.0.0.0", port=port)
