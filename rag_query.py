from dotenv import load_dotenv
from langfuse import Langfuse
from web_search_tool import web_search
from langgraph.graph import StateGraph, START, END
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from typing import TypedDict, List, Optional
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


COLLECTION = "python_docs"
LLM = "llama3.2:3b"
SCORE_THRESHOLD = 0.55
NOT_FOUND = "NOT_IN_CONTEXT"

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
client = MilvusClient(uri="http://localhost:19530")


class RAGState(TypedDict):
    question: str
    context: List[str]
    sources: List[str]
    top_score: float
    web_results: str
    route: str
    answer: str
    # may be None (API) or a Langfuse trace (terminal)
    trace: Optional[object]


def _span(state, name, inp):
    """Start a Langfuse span only if a trace exists; otherwise return None."""
    trace = state.get("trace")
    return trace.span(name=name, input=inp) if trace else None


def _end(span, output):
    if span:
        span.end(output=output)


def retrieve(state: RAGState) -> RAGState:
    span = _span(state, "retrieve", state["question"])
    q_vector = embedding_model.embed_query(state["question"])
    hits = client.search(
        collection_name=COLLECTION,
        data=[q_vector],
        limit=3,
        output_fields=["text", "source", "page"],
    )[0]
    chunks = [h["entity"]["text"] for h in hits]
    sources = [
        f"{h['entity']['source']} (page {h['entity']['page']})" for h in hits]
    top_score = hits[0]["distance"] if hits else 0.0
    _end(span, {"sources": sources, "top_score": top_score})
    return {"context": chunks, "sources": sources, "top_score": top_score}


def score_gate(state: RAGState) -> RAGState:
    score = state["top_score"]
    route = "docs" if score >= SCORE_THRESHOLD else "web"
    trace = state.get("trace")
    if trace:
        trace.event(
            name="routing_decision",
            input={"top_score": round(score, 3), "threshold": SCORE_THRESHOLD},
            output={"route": route},
        )
    print(
        f"\n[Routing] top document score = {score:.3f} (threshold {SCORE_THRESHOLD})  ->  trying {route.upper()}")
    return {"route": route}


def gate_decision(state: RAGState) -> str:
    return state["route"]


def try_docs(state: RAGState) -> RAGState:
    gen = _span(state, "generate_from_docs", state["question"])
    context = "\n\n".join(state["context"])
    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, reply with EXACTLY this and nothing else: {NOT_FOUND}

Context:
{context}

Question: {state["question"]}
Answer:"""
    response = ollama.chat(model=LLM, messages=[
                           {"role": "user", "content": prompt}])
    answer = response["message"]["content"].strip()

    if NOT_FOUND in answer:
        _end(gen, "NOT_IN_CONTEXT -> falling back to web")
        print("[Docs] Answer not found in documents -> falling back to WEB")
        return {"route": "web", "answer": ""}

    _end(gen, answer)
    print("\n--- Answered from DOCUMENTS ---")
    for s in state["sources"]:
        print(f"  {s}")
    print("-------------------------------")
    return {"route": "docs_done", "answer": answer}


def after_docs(state: RAGState) -> str:
    return "web" if state["route"] == "web" else "done"


def search_web(state: RAGState) -> RAGState:
    span = _span(state, "web_search", state["question"])
    print("\n[Web search] Searching the web...")
    results = web_search(state["question"])
    _end(span, results[:500])
    return {"web_results": results}


def generate_from_web(state: RAGState) -> RAGState:
    gen = _span(state, "generate_from_web", state["question"])
    prompt = f"""Answer the question using ONLY the web results below.

Web results:
{state["web_results"]}

Question: {state["question"]}
Answer:"""
    response = ollama.chat(model=LLM, messages=[
                           {"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    _end(gen, answer)
    print("\n--- Answered from WEB ---")
    for line in state["web_results"].split("\n"):
        if line.strip().startswith("(http"):
            print(f"  {line.strip()}")
    print("-------------------------")
    return {"answer": answer}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("score_gate", score_gate)
builder.add_node("try_docs", try_docs)
builder.add_node("search_web", search_web)
builder.add_node("generate_from_web", generate_from_web)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "score_gate")
builder.add_conditional_edges(
    "score_gate", gate_decision,
    {"docs": "try_docs", "web": "search_web"},
)
builder.add_conditional_edges(
    "try_docs", after_docs,
    {"done": END, "web": "search_web"},
)
builder.add_edge("search_web", "generate_from_web")
builder.add_edge("generate_from_web", END)

graph = builder.compile()


if __name__ == "__main__":
    print("Agentic RAG ready (with Langfuse tracing).\n")
    while True:
        q = input("Ask a question (or 'quit'): ")
        if q.strip().strip("'\"").lower() == "quit":
            break
        trace = langfuse.trace(name="rag_query", input=q)
        result = graph.invoke({"question": q, "trace": trace})
        trace.update(output=result["answer"])
        langfuse.flush()
        print("\n" + result["answer"] + "\n")
