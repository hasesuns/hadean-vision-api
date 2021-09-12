import io
import os
import shutil
from logging import DEBUG, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.logger import logger as fastapi_logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from hadeanvision import ConvertParams, convert

gunicorn_logger = getLogger("gunicorn.error")
fastapi_logger.handlers = gunicorn_logger.handlers
fastapi_logger.setLevel(DEBUG)

load_dotenv()


class RequestConvertParam(BaseModel):
    num_colors: int
    rgb_list: List[Tuple[float, float, float]] = []
    bgr_list: List[Tuple[float, float, float]] = []


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/convert/",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Return image/png in bytes format.",
        }
    },
)
async def hadean_convert(file: UploadFile = File(...)):
    fastapi_logger.debug(file)
    file_path = save_upload_file_tmp(file)
    try:
        img = cv2.imread(str(file_path))
        img_converted = convert(img)
        _, img_png = cv2.imencode(".png", img_converted)
        return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to convert an image.")
    finally:
        file_path.unlink()


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


if os.environ.get("IS_DEBUG"):

    @app.get("/")
    def read_root():
        fastapi_logger.debug("access!")
        return {"Hello": "World"}

    @app.get("/debug/", response_class=HTMLResponse)
    async def main():
        content = """
            <body>
              <form action="/convert/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" multiple>
                <input type="submit">
              </form>
            </body>
        """
        return HTMLResponse(content=content)
