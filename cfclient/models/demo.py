from typing import Any
from pydantic import Field
from pydantic import BaseModel

from .core import AlgorithmBase


demo_hello_endpoint = "/demo/hello"


class HelloModel(BaseModel):
    name: str = Field(..., description="Name that you want to say Hello to!")


class HelloResponse(BaseModel):
    msg: str = Field(..., description="The generated Hello message!")


@AlgorithmBase.register("demo.hello")
class Hello(AlgorithmBase):
    endpoint = demo_hello_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: HelloModel, *args: Any) -> HelloResponse:
        return HelloResponse(msg=f"Hello, {data.name}!")


__all__ = [
    "demo_hello_endpoint",
    "Hello",
    "HelloModel",
    "HelloResponse",
]
