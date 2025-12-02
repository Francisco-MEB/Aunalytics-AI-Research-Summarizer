from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router_scrape import router as scrape_router
from backend.api.router_ingest import router as ingest_router
from backend.api.router_chat import router as chat_router
from backend.api.router_query import router as query_router
from backend.api.router_health import router as health_router

app = FastAPI(title="RAPTOR Research Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health")
app.include_router(ingest_router, prefix="/ingest")
app.include_router(query_router, prefix="/query")
app.include_router(chat_router, prefix="/chat")
app.include_router(scrape_router, prefix="/scrape")

@app.get("/")
def root():
    return {"message": "RAPTOR backend operational"}

print("Routers loaded:", app.routes)
