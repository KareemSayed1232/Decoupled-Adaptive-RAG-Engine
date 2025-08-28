
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from contextlib import asynccontextmanager
import json
from .orchestrator import initialize_rag_system, get_rag_orchestrator
from .utils import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RAG-API: Initializing orchestrator...")
    await initialize_rag_system()
    print("RAG-API: System ready.")
    yield

app = FastAPI(title="RAG API", lifespan=lifespan)

@app.middleware("http")
async def connection_reset_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except ConnectionResetError:
        logger.info("Client disconnected. (ConnectionResetError)")

        return JSONResponse(status_code=204) 

@app.post("/ask")
async def ask(request: dict):
    question = request.get("question", "")
    orchestrator = get_rag_orchestrator()

    from fastapi.responses import StreamingResponse
    async def stream_wrapper():
        try:
            async for event in orchestrator.ask_question(question):
                yield f"data: {json.dumps(event)}\n\n"
        except ConnectionResetError:
 
            logger.info("Client disconnected after stream completion. (ConnectionResetError)")
        except Exception as e:
            logger.error(f"An unexpected error occurred during streaming: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': 'A server error occurred.'})}\n\n"
            
    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
