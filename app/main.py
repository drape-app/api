from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.garments import router as garments_router
from app.api.wardrobe import router as wardrobe_router
from app.api.outfits import router as outfits_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up embedding model in background
    from app.services.embedding import _load_model
    _load_model()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Drape API",
    version="0.1.0",
    description="Cloud backend for the Drape digital wardrobe app",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(garments_router)
app.include_router(wardrobe_router)
app.include_router(outfits_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "drape-api"}
