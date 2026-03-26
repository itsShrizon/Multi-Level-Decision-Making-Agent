"""HTTP response helpers — a single envelope shape for every endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def ok(data: Any, status_code: int = 200, **metadata: Any) -> JSONResponse:
    body: dict[str, Any] = {"success": True, "data": data}
    if metadata:
        body["metadata"] = metadata
    return JSONResponse(status_code=status_code, content=body)
