"""
SQLite 数据库操作模块
用于存储设备信息和事件日志
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

DB_FILE = "device_logs.db"
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_FILE, timeout=10)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT UNIQUE NOT NULL,
            name TEXT DEFAULT '',
            location TEXT DEFAULT '',
            last_seen TEXT NOT NULL,
            status TEXT DEFAULT 'online',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_category TEXT NOT NULL,
            name TEXT DEFAULT '',
            score REAL DEFAULT 0.0,
            message TEXT DEFAULT '',
            details TEXT DEFAULT '{}',
            device_time TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'success',
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_device_id ON events(device_id);
        CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
    """)
    conn.commit()


def upsert_device(device_id: str, name: str = ""):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing = conn.execute(
        "SELECT device_id, name FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO devices (device_id, name, last_seen, status, created_at) VALUES (?, ?, ?, 'online', ?)",
            (device_id, name or device_id, now, now),
        )
    else:
        conn.execute(
            "UPDATE devices SET last_seen = ?, status = 'online' WHERE device_id = ?",
            (now, device_id),
        )
        if name and not existing["name"]:
            conn.execute("UPDATE devices SET name = ? WHERE device_id = ?", (name, device_id))
    conn.commit()


def update_device_status():
    conn = get_connection()
    threshold = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE devices SET status = 'offline' WHERE last_seen < ? AND status = 'online'",
        (threshold,),
    )
    conn.commit()


EVENT_MAP = {
    "system_start": ("系统启动", "event", "success"),
    "access_granted": ("人员进出", "event", "success"),
    "door_close": ("门关闭", "event", "success"),
    "register": ("人脸注册", "event", "success"),
    "delete": ("人脸删除", "event", "success"),
    "liveness_fail": ("活体检测失败", "event", "abnormal"),
    "stranger": ("陌生人报警", "event", "abnormal"),
    "open": ("门打开", "door", "success"),
}




def _map_event(raw_event: str, message: str = "") -> tuple:
    if raw_event in EVENT_MAP:
        return EVENT_MAP[raw_event]
    return ("未知事件", "event", "success")


def _build_details(data: dict, event_type: str) -> str:
    if event_type == "access_granted":
        return f"用户: {data.get('name', '')} | 分数: {data.get('score', 0):.2f} | 开门"
    elif event_type == "door_close":
        return f"用户: {data.get('name', '')} | 门已关闭"
    elif event_type == "liveness_fail":
        return f"用户: {data.get('name', '')} | 原因: {data.get('message', '')}"
    elif event_type == "stranger":
        return "连续检测到陌生人 | 已抓拍"
    elif event_type == "register":
        msg = data.get("message", "")
        return "注册成功" if msg == "success" else "注册失败/取消"
    elif event_type == "delete":
        msg = data.get("message", "")
        if msg == "database_empty":
            return "数据库为空"
        return "删除成功" if msg == "success" else "删除失败/取消"
    elif event_type == "system_start":
        return data.get("message", "系统启动")
    elif event_type == "open":
        return f"用户: {data.get('name', '')} | 门已打开"
    return data.get("message", "")


def insert_event(device_id: str, data: dict):
    msg_type = data.get("type", "event")
    if msg_type == "performance":
        return

    conn = get_connection()
    raw_event = data.get("event", "")
    msg = data.get("message", "")
    display_name, category, status = _map_event(raw_event, msg)
    details = _build_details(data, raw_event)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_time = data.get("time", now)
    person_name = data.get("name", "")
    score = data.get("score", 0.0)

    conn.execute(
        """INSERT INTO events
           (device_id, event_type, event_category, name, score, message, details, device_time, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (device_id, display_name, category, person_name, score, msg, details, device_time, now, status),
    )
    conn.commit()


def get_dashboard_stats() -> dict:
    conn = get_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    today_traffic = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='人员进出' AND created_at >= ?",
        (today,),
    ).fetchone()[0]

    yesterday_traffic = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='人员进出' AND created_at >= ? AND created_at < ?",
        (yesterday, today),
    ).fetchone()[0]

    if yesterday_traffic > 0:
        traffic_trend = round((today_traffic - yesterday_traffic) / yesterday_traffic * 100, 1)
    else:
        traffic_trend = 0.0

    update_device_status()
    total_devices = conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
    online_devices = conn.execute("SELECT COUNT(*) FROM devices WHERE status='online'").fetchone()[0]

    liveness_total = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type IN ('人员进出', '活体检测失败') AND created_at >= ?",
        (today,),
    ).fetchone()[0]
    liveness_passed = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='人员进出' AND created_at >= ?",
        (today,),
    ).fetchone()[0]
    liveness_rate = round(liveness_passed / liveness_total * 100, 1) if liveness_total > 0 else 100.0

    yesterday_liveness_total = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type IN ('人员进出', '活体检测失败') AND created_at >= ? AND created_at < ?",
        (yesterday, today),
    ).fetchone()[0]
    yesterday_liveness_passed = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='人员进出' AND created_at >= ? AND created_at < ?",
        (yesterday, today),
    ).fetchone()[0]
    if yesterday_liveness_total > 0:
        yesterday_rate = round(yesterday_liveness_passed / yesterday_liveness_total * 100, 1)
        liveness_trend = round(liveness_rate - yesterday_rate, 1)
    else:
        liveness_trend = 0.0

    stranger_alerts = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='陌生人报警' AND created_at >= ?",
        (today,),
    ).fetchone()[0]

    return {
        "today_traffic": today_traffic,
        "traffic_trend": traffic_trend,
        "online_devices": online_devices,
        "total_devices": total_devices,
        "liveness_rate": liveness_rate,
        "liveness_trend": liveness_trend,
        "stranger_alerts": stranger_alerts,
    }


def get_events(page: int = 1, per_page: int = 10, device_id: str = "",
               status: str = "", date: str = "") -> dict:
    conn = get_connection()
    where = []
    params = []

    if device_id:
        where.append("e.device_id = ?")
        params.append(device_id)
    if status:
        where.append("e.status = ?")
        params.append(status)
    if date:
        where.append("e.created_at >= ?")
        params.append(date)
        where.append("e.created_at < ?")
        params.append(date + " 23:59:59")

    where_sql = " AND ".join(where) if where else "1=1"

    total = conn.execute(
        f"SELECT COUNT(*) FROM events e WHERE {where_sql}", params
    ).fetchone()[0]

    offset = (page - 1) * per_page
    rows = conn.execute(
        f"""SELECT e.*, d.name as device_name
            FROM events e
            LEFT JOIN devices d ON e.device_id = d.device_id
            WHERE {where_sql}
            ORDER BY e.id DESC
            LIMIT ? OFFSET ?""",
        params + [per_page, offset],
    ).fetchall()

    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "device_id": row["device_id"],
            "device_name": row["device_name"] or row["device_id"],
            "event_type": row["event_type"],
            "event_category": row["event_category"],
            "details": row["details"],
            "time": row["device_time"] or row["created_at"],
            "status": row["status"],
        })

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "items": items,
    }


def get_devices() -> list:
    conn = get_connection()
    update_device_status()
    rows = conn.execute(
        "SELECT * FROM devices ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def update_device_name(device_id: str, name: str):
    conn = get_connection()
    conn.execute("UPDATE devices SET name = ? WHERE device_id = ?", (name, device_id))
    conn.commit()


def export_csv(device_id: str = "", status: str = "", date: str = "") -> str:
    data = get_events(page=1, per_page=100000, device_id=device_id, status=status, date=date)
    lines = ["设备ID,设备名称,事件类型,详情,上传时间,状态"]
    for item in data["items"]:
        line = ','.join([
            item["device_id"],
            item["device_name"],
            item["event_type"],
            item["details"].replace(",", "，"),
            item["time"],
            "成功" if item["status"] == "success" else "异常",
        ])
        lines.append(line)
    return "\n".join(lines)
