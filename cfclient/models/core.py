import time
import logging

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from fastapi import File
from pydantic import Field
from pydantic import BaseModel
from pydantic import validator

from ..core import HttpClient
from ..core import TritonClient
from ..utils import log_times
from ..utils import log_endpoint
from ..utils import distances2scores


# common requests

upload_file_field = File(..., description="> limit size: 20mb")

class RetrievalModel(BaseModel):
    top_k: int = Field(
        ...,
        ge=1,
        description="""
Number of candidates that we want to retrieve.
> Notice that the response may not always contain `top_k` candidates if `num_probe` is not large enough (see `num_probe` below for more details).
""",
    )
    num_probe: int = Field(
        16,
        description="""
The `num_probe` parameter used by [faiss](https://github.com/facebookresearch/faiss).
+ Basically, the larger this number is, the larger the `candidate pool` will be (see `num_candidates` below for more details).
+ If the size of the `candidate pool` smaller than `top_k`, then the response will only contain all the candidates from the `candidate pool`.
> Which means in this case, the number of the candidates in the response will be smaller than `top_k`.
        """,
    )
    num_candidates: int = Field(
        10 ** 4,
        description="""
Number of candidates that we want to have in our `candidate pool`. It should be at least ≥ `top_k`, and in most cases, should > `top_k`.
> Notice that if `num_probe` / our database is not large enough, the actual size of the `candidate pool` might be smaller than `num_candidates`.

This is useful when the `top_k` candidates does not always serve our requirements well, for example:
+ We want to search for the best 16 models in our database.
+ We don't want any two of these models belong to the same group.

In this case, we may need to search for more than 16 models, which will lead to a `candidate pool` which contains more than 16 candidates.

> To sum up, in order to achieve various kinds of goals, we will first collect a large `candidate pool`, then pick up the `top_k` candidates that satisfy our actual needs.
        """,
    )

    @validator("num_candidates")
    def ensure_num_candidates_ge_top_k(cls, v: int, values: Dict[str, Any]) -> int:
        top_k = values.get("top_k")
        if top_k is not None and v < top_k:
            raise ValueError("`num_candidates` should be ≥ `top_k` when it is not `0` (`0` will trigger default behaviours)")
        return v

class ImageModel(BaseModel):
    url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the user's image.
> `cos` url from `qcloud` is preferred.
"""
    )

class ImageRetrievalModel(RetrievalModel, ImageModel):
    pass

class TextModel(BaseModel):
    text: str = Field(..., description="The text that we want to handle.")

class TextRetrievalModel(RetrievalModel, TextModel):
    pass

class RetrievalResponse(BaseModel):
    distances: List[float] = Field(..., description="Distances of the retrieved candidates.")
    scores: List[float] = Field(
        ...,
        description="""
Scores of the retrieved candidates.
+ These scores will all range from `0` to `1`.
+ It is already sorted in the descending order.
+ The best score will always be `1` (which means `scores[0]` will always be `1`).
""",
    )

class PosterCodesResponse(BaseModel):
    group_codes: List[str] = Field(..., description="Group codes of the retrieved candidates.")
    model_codes: List[str] = Field(..., description="Model codes of the retrieved candidates.")


# algorithms

def register_core(
    name: str,
    global_dict: Dict[str, type],
    *,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
):
    def _register(cls):
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None:
            print(
                f"> [warning] '{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


T = TypeVar("T", bound="WithRegister", covariant=True)


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        def before(cls_: Type[T]) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


algorithms = {}


class AlgorithmBase(WithRegister, metaclass=ABCMeta):
    d = algorithms
    endpoint: str

    def __init__(self, clients: Dict[str, Any]) -> None:
        self.clients = clients

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    async def run(self, data: Any, *args: Any) -> Any:
        pass

    @property
    def http_client(self) -> Optional[HttpClient]:
        return self.clients.get("triton")

    @property
    def triton_client(self) -> Optional[TritonClient]:
        return self.clients.get("triton")

    def log_endpoint(self, data: BaseModel) -> None:
        log_endpoint(self.endpoint, data)
    
    def log_times(self, latencies: Dict[str, float]) -> None:
        log_times(self.endpoint, latencies)


def retrieval_post_process(
    *,
    endpoint: str,
    data: RetrievalModel,
    latencies: Dict[str, float],
    distances: List[float],
    id_key: Union[str, List[str]],
    t: float,
    **results: List[Any],
) -> Dict[str, Any]:
    sorted_indices = np.argsort(distances)
    final_results = {k: [] for k in results}
    final_distances = []
    added_ids = set()
    for index in sorted_indices:
        if isinstance(id_key, str):
            local_id = results[id_key][index]
        else:
            local_id = tuple(results[k][index] for k in id_key)
        if local_id in added_ids:
            continue
        added_ids.add(local_id)
        for k, v in results.items():
            final_results[k].append(v[index])
        final_distances.append(distances[index])
        if len(added_ids) == data.num_candidates:
            break
    final_scores = distances2scores(final_distances)

    num_final = len(added_ids)
    top_k_req = data.top_k
    if top_k_req > num_final:
        logging.warning(f"only {num_final} candidates are available, but top_k={top_k_req} is requested")
        top_k_req = num_final

    latencies["post_process"] = time.time() - t
    log_times(endpoint, latencies)

    final_results["distances"] = final_distances
    final_results["scores"] = final_scores
    final_results = {k: v[:top_k_req] for k, v in final_results.items()}
    return final_results


__all__ = [
    "upload_file_field",
    "RetrievalModel",
    "ImageModel",
    "ImageRetrievalModel",
    "TextModel",
    "TextRetrievalModel",
    "RetrievalResponse",
    "PosterCodesResponse",
    "algorithms",
    "AlgorithmBase",
    "retrieval_post_process",
]
