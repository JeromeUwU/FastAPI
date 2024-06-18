from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Load the model and processor once
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate/")
async def translate(request: TranslationRequest):
    try:
        text_inputs = processor(text=request.text, src_lang=request.src_lang, return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang=request.tgt_lang, generate_speech=False)
        translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
