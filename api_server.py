import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# reuse your existing agent AND the langfuse client from rag_query
from rag_query import graph, langfuse

MODEL_NAME = "rag-agent"


# flush Langfuse gracefully when the server shuts down (not per-request)
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()   # send any remaining traces on shutdown

app = FastAPI(title="RAG API", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    question = req.messages[-1].content

    # create a trace for this web-UI question (NO flush here - background sends it)
    trace = langfuse.trace(name="rag_query_webui", input=question)
    print(f"[Langfuse] trace created: {trace.id}")   # debug line
    result = graph.invoke({"question": question, "trace": trace})
    answer = result["answer"]
    trace.update(output=answer)
    langfuse.flush()                                 # force-send for now (debug)
    print("[Langfuse] flushed")                      # debug line
    # note: no langfuse.flush() here - that was the cause of the error

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {"index": 0, "message": {"role": "assistant",
                                     "content": answer}, "finish_reason": "stop"},
        ],
    }


@app.get("/")
def root():
    return {"status": "RAG API running"}
