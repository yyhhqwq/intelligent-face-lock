"""
K230 人脸识别门禁系统 - 数据上报接收服务器
FastAPI 实现，支持设备管理、事件存储、统计查询
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import database as db

HOST = "0.0.0.0"
PORT = 8080
LOG_FILE = "upload.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("k230-server")

app = FastAPI(title="K230 Device Log Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    db.init_db()
    logger.info(f"服务器启动 → http://{HOST}:{PORT}")
    logger.info(f"数据库 → {Path(db.DB_FILE).absolute()}")


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.api_route("/api/upload", methods=["POST", "PUT", "PATCH"])
async def upload(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid json"})

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(body, ensure_ascii=False)
    print(f"[RECV] {timestamp} {line}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    msg_type = body.get("type", "event")
    device_id = body.get("device_id", "")
    if not device_id:
        device_id = request.headers.get("X-Device-ID", "unknown")

    db.upsert_device(device_id, body.get("device_name", ""))
    db.insert_event(device_id, body)

    return {"status": "ok"}


@app.get("/api/dashboard")
async def dashboard():
    return db.get_dashboard_stats()


@app.get("/api/events")
async def events(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    device_id: str = Query(""),
    status: str = Query(""),
    date: str = Query(""),
):
    return db.get_events(page, per_page, device_id, status, date)


@app.get("/api/devices")
async def devices():
    return db.get_devices()


@app.put("/api/devices/{device_id}")
async def update_device(device_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid json"})

    name = body.get("name", "")
    db.update_device_name(device_id, name)
    return {"status": "ok"}


@app.get("/api/export")
async def export_csv(
    device_id: str = Query(""),
    status: str = Query(""),
    date: str = Query(""),
):
    csv_content = db.export_csv(device_id, status, date)
    filename = f"device_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter(["\ufeff" + csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
