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
from typing import TypeVar
from typing import Callable
from typing import Awaitable
from aiohttp import ClientSession
from pydantic import BaseModel
from cftool.web import raise_err


async def get(url: str, session: ClientSession, **kwargs: Any) -> bytes:
    async with session.get(url, **kwargs) as response:
        return await response.read()


async def post(
    url: str,
    json: Dict[str, Any],
    session: ClientSession,
    **kwargs: Any,
) -> Dict[str, Any]:
    async with session.post(url, json=json, **kwargs) as response:
        return await response.json()


def log_endpoint(endpoint: str, data: BaseModel) -> None:
    msg = f"{endpoint} endpoint entered with kwargs : {json.dumps(data.dict(), ensure_ascii=False)}"
    logging.debug(msg)


def log_times(endpoint: str, times: Dict[str, float]) -> None:
    times["__total__"] = sum(times.values())
    logging.debug(f"elapsed time of endpoint {endpoint} : {json.dumps(times)}")


async def run_algorithm(algorithm: Any, data: BaseModel, *args: Any) -> BaseModel:
    try:
        return await algorithm.run(data, *args)
    except Exception as err:
        raise_err(err)


async def _download(session: ClientSession, url: str) -> bytes:
    try:
        return await get(url, session)
    except Exception:
        return requests.get(url).content


async def _download_image(session: ClientSession, url: str) -> Image.Image:
    raw_data = None
    try:
        raw_data = await _download(session, url)
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


TRes = TypeVar("TRes")


async def _download_with_retry(
    download_fn: Callable[[ClientSession, str], Awaitable[TRes]],
    session: ClientSession,
    url: str,
    retry: int = 3,
    interval: int = 1,
) -> TRes:
    msg = ""
    for i in range(retry):
        try:
            res = await download_fn(session, url)
            if i > 0:
                logging.warning(f"succeeded after {i} retries")
            return res
        except Exception as err:
            msg = str(err)
        time.sleep(interval)
    raise ValueError(f"{msg}\n(After {retry} retries)")


async def download_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = 3,
    interval: int = 1,
) -> bytes:
    return await _download_with_retry(_download, session, url, retry, interval)


async def download_image_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = 3,
    interval: int = 1,
) -> Image.Image:
    return await _download_with_retry(_download_image, session, url, retry, interval)


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
    "log_times",
    "log_endpoint",
    "download_image",
    "download_image_with_retry",
    "distances2scores",
]
