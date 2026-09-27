from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback

from metrics import build_dashboard

BASE = Path(__file__).resolve().parent

app = FastAPI(
    title="股票基本面驾驶舱",
    version="1.0"
)

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)


@app.get("/")
def index():
    response = FileResponse(
        BASE / "static" / "index.html"
    )
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate"
    )
    return response


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "stock-fundamental-dashboard",
        "version": "2026-09-27-v2"
    }


@app.get("/api/stock/{symbol}")
def stock(symbol: str):
    symbol = symbol.strip()

    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="请输入股票代码"
        )

    try:
        return build_dashboard(symbol)

    except Exception as exc:
        error_text = traceback.format_exc()

        print("========== STOCK ERROR ==========")
        print(error_text)
        print("=================================")

        raise HTTPException(
            status_code=502,
            detail=f"数据获取失败：{exc}"
        )
