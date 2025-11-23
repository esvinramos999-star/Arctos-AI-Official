from fastapi import FastAPI
from pydantic import BaseModel
from inference_engine import ArctosEngine

app = FastAPI()
engine = ArctosEngine()

class Query(BaseModel):
    prompt: str

@app.post("/arctos")
def arctos_answer(query: Query):
    response = engine.generate(query.prompt)
    return {"response": response}
