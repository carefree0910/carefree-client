import sys
import json
import time
import logging
import requests
import logging.config

import numpy as np

from io import BytesIO
from PIL import Image
from PIL import ImageOps
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Optional
from aiohttp import ClientSession
from fastapi import Response
from fastapi import HTTPException
from pydantic import BaseModel


async def get(url: str, session: ClientSession) -> bytes:
    async with session.get(url) as response:
        return await response.read()


async def post(
    url: str,
    json: Dict[str, Any],
    session: ClientSession,
) -> Dict[str, Any]:
    async with session.post(url, json=json) as response:
        return await response.json()


error_code = 406


class RuntimeError(BaseModel):
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "RuntimeError occurred."},
        }


def get_err_msg(err: Exception) -> str:
    return " | ".join(map(repr, sys.exc_info()[:2] + (str(err),)))


def raise_err(err: Exception) -> None:
    logging.exception(err)
    raise HTTPException(status_code=error_code, detail=get_err_msg(err))


def log_endpoint(endpoint: str, data: BaseModel) -> None:
    msg = f"{endpoint} endpoint entered with kwargs : {json.dumps(data.dict(), ensure_ascii=False)}"
    logging.debug(msg)


def log_times(endpoint: str, times: Dict[str, float]) -> None:
    times["__total__"] = sum(times.values())
    logging.debug(f"elapsed time of endpoint {endpoint} : {json.dumps(times)}")


def get_responses(
    success_model: Type[BaseModel],
    *,
    json_example: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Type]]:
    success_response = {"model": success_model}
    if json_example is not None:
        content = success_response["content"] = {}
        json_field = content["application/json"] = {}
        json_field["example"] = json_example
    return {
        200: success_response,
        error_code: {"model": RuntimeError},
    }


async def run_algorithm(algorithm: Any, data: BaseModel, *args: Any) -> BaseModel:
    try:
        return await algorithm.run(data, *args)
    except Exception as err:
        raise_err(err)


def get_image_response_kwargs() -> Dict[str, Any]:
    example = "\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x00\\x00\\x00\\x00:~\\x9bU\\x00\\x00\\x00\\nIDATx\\x9cc`\\x00\\x00\\x00\\x02\\x00\\x01H\\xaf\\xa4q\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"
    responses = {
        200: {"content": {"image/png": {"example": example}}},
        error_code: {"model": RuntimeError},
    }
    description = """
Bytes of the output image.
+ When using `requests` in `Python`, you can get the `bytes` with `res.content`.
+ When using `fetch` in `JavaScript`, you can get the `Blob` with `await res.blob()`.
"""
    return dict(
        responses=responses,
        response_class=Response(content=b""),
        response_description=description,
    )


async def _download(session: ClientSession, url: str) -> bytes:
    try:
        return await get(url, session)
    except Exception:
        return requests.get(url).content


async def _download_image(session: ClientSession, url: str) -> Image.Image:
    raw_data = None
    try:
        raw_data = await _download(url, session)
        return Image.open(BytesIO(raw_data))
    except Exception as err:
        if raw_data is None:
            msg = f"raw | None | err | {err}"
        else:
            try:
                msg = raw_data.decode("utf-8")
            except:
                msg = f"raw | {raw_data[:20]} | err | {err}"
        raise ValueError(msg)


async def download_image(session: ClientSession, url: str) -> Image.Image:
    img = await _download_image(session, url)
    img = ImageOps.exif_transpose(img)
    return img


async def download_image_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = 3,
    interval: int = 1,
) -> Image.Image:
    msg = ""
    for i in range(retry):
        try:
            image = await download_image(session, url)
            if i > 0:
                logging.warning(f"succeeded after {i} retries")
            return image
        except Exception as err:
            msg = str(err)
        time.sleep(interval)
    raise ValueError(f"{msg}\n(After {retry} retries)")


def distances2scores(distances: List[float]) -> List[float]:
    distances_arr = np.array(distances, np.float32)
    if len(distances_arr) == 1:
        scores_arr = np.array([1.0], np.float32)
    else:
        dmin, dmax = distances_arr[0], distances_arr[-1]
        scores_arr = 1.0 - (distances_arr - dmin) / (dmax - dmin + 1.0e-6)
    return scores_arr.tolist()


__all__ = [
    "get",
    "post",
    "raise_err",
    "log_times",
    "get_err_msg",
    "log_endpoint",
    "get_responses",
    "download_image",
    "download_image_with_retry",
    "distances2scores",
]
