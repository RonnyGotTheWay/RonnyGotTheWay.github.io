from fastapi import Request
from fastapi.responses import JSONResponse


async def unhandled_error(_: Request, exception: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal error", "error_type": type(exception).__name__})
