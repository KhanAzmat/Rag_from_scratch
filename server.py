'''
Running a server allows the resource-intensive tasks to be done only once, on the server start.
'''
from fastapi import FastAPI
from pydantic import BaseModel
from rag_implementation import RagFromScratch
import logging

app = FastAPI(
    title='Rag-from-scratch Service',
    version='1.0.0'
)

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

pdf_path = './Rag-From-Scratch/simple-local-rag/human-nutrition-text.pdf'
embedding_model = 'all-mpnet-base-v2'
reranking_model = 'mixedbread-ai/mxbai-rerank-large-v1'

rag_from_scratch = RagFromScratch(embedding_model, reranking_model, pdf_path)

@app.on_event('startup')
def startup():
    # called once when the server starts and loads the models and creates the vector store.
    print(f'[INFO] Server startup:Building pipeline')
    rag_from_scratch.run()
    print(f'[INFO] Pipeline ready')


class QueryRequest(BaseModel):
    query: str

@app.post('/ask')
def ask_llm(request:QueryRequest):
    answer, contexts = rag_from_scratch.ask(request.query)
    print(f'Context : {contexts}')
    return answer

