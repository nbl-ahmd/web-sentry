from fastapi import FastAPI
from pydantic import BaseModel
from model import test_url, check_homograph

app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/check-url")
async def check_url(request: URLRequest):
    return check_homograph(request.url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1,
        timeout_keep_alive=75,
        timeout_notify=30,
        timeout_graceful_shutdown=10,
        log_level="info",
        limit_concurrency=10,
        backlog=128,
        limit_max_requests=0,
        timeout=30)
