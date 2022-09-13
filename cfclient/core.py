import gzip
import json
import zlib
import struct
import gevent.pool

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from aiohttp import TCPConnector
from aiohttp import ClientSession
from geventhttpclient import HTTPClient
from urllib.parse import quote
from urllib.parse import quote_plus
from geventhttpclient.url import URL
from geventhttpclient.response import HTTPSocketPoolResponse as Response


def np_to_triton_dtype(np_dtype: np.dtype) -> Optional[str]:
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def triton_to_np_dtype(dtype: str) -> np.dtype:
    if dtype == "BOOL":
        return bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.object_
    return None


def serialize_byte_tensor(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr.size == 0:
        return np.empty([0], dtype=np.object_)
    if (arr.dtype == np.object_) or (arr.dtype.type == np.bytes_):
        flattened_ls = []
        for obj in np.nditer(arr, flags=["refs_ok"], order="C"):
            if arr.dtype == np.object_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = str(obj.item()).encode("utf-8")
            else:
                s = obj.item()
            flattened_ls.append(struct.pack("<I", len(s)))
            flattened_ls.append(s)
        flattened = b"".join(flattened_ls)
        flattened_array = np.asarray(flattened, dtype=np.object_)
        if not flattened_array.flags["C_CONTIGUOUS"]:
            flattened_array = np.ascontiguousarray(flattened_array, dtype=np.object_)
        return flattened_array
    else:
        raise_error("cannot serialize bytes tensor: invalid datatype")
    return None


class InputData:
    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self._datatype = np_to_triton_dtype(data.dtype)
        if self._datatype != "BYTES":
            raw_data = data.tobytes()
        else:
            serialized_output = serialize_byte_tensor(data)
            assert serialized_output is not None
            raw_data = serialized_output.item() if serialized_output.size > 0 else b""
        self._raw_data = raw_data

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.data.shape,
            "datatype": self._datatype,
            "parameters": {"binary_data_size": len(self._raw_data)},
        }


def get_request(
    inputs: List[InputData],
    priority: int = 0,
    timeout: Optional[int] = None,
    outputs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bytes, int]:
    parameters: Dict[str, Any] = {"binary_data_output": True}
    if priority != 0:
        parameters["priority"] = priority
    if timeout is not None:
        parameters["timeout"] = timeout
    request = {
        "inputs": [inp.info for inp in inputs],
        "parameters": parameters,
    }
    if outputs is not None:
        request["outputs"] = outputs
    request_body = json.dumps(request)
    binary_data = None
    for inp in inputs:
        if binary_data is None:
            binary_data = inp._raw_data
        else:
            binary_data += inp._raw_data
    assert binary_data is not None

    json_size = len(request_body)
    request_body_bytes = struct.pack(
        f"{json_size}s{len(binary_data)}s",
        request_body.encode(),
        binary_data,
    )
    return request_body_bytes, json_size


def _get_query_string(query_params: Dict[str, Any]) -> str:
    params = []
    for key, value in query_params.items():
        if not isinstance(value, list):
            params.append("%s=%s" % (quote_plus(key), quote_plus(str(value))))
        else:
            for item in value:
                params.append("%s=%s" % (quote_plus(key), quote_plus(str(item))))
    if params:
        return "&".join(params)
    return ""


class InferenceServerException(Exception):
    def __init__(
        self,
        msg: str,
        status: Optional[str] = None,
        debug_details: Optional[str] = None,
    ):
        self._msg = msg
        self._status = status
        self._debug_details = debug_details

    def __str__(self) -> str:
        msg = super().__str__() if self._msg is None else self._msg
        if self._status is not None:
            msg = "[" + self._status + "] " + msg
        return msg

    def message(self) -> str:
        return self._msg

    def status(self) -> Optional[str]:
        return self._status

    def debug_details(self) -> Optional[str]:
        return self._debug_details


def _get_error(response: Response) -> Optional[InferenceServerException]:
    if response.status_code == 200:
        return None
    error_response = json.loads(response.read())
    return InferenceServerException(msg=error_response["error"])


def _raise_if_error(response: Response) -> None:
    error = _get_error(response)
    if error is not None:
        raise error


def raise_error(msg: str) -> None:
    raise InferenceServerException(msg=msg)


def deserialize_bytes_tensor(encoded_tensor: Any) -> Any:
    strings = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        length = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(length), val_buf, offset)[0]
        offset += length
        strings.append(sb)
    return np.array(strings, dtype=np.object_)


class InferResult:
    def __init__(self, response: Response, verbose: bool):
        header_length = response.get("Inference-Header-Content-Length")

        class DecompressedResponse:
            def __init__(self, decompressed_data: Any):
                self.decompressed_data_ = decompressed_data
                self.offset_ = 0

            def read(self, length: int = -1) -> Any:
                if length == -1:
                    return self.decompressed_data_[self.offset_ :]
                else:
                    prev_offset = self.offset_
                    self.offset_ += length
                    return self.decompressed_data_[prev_offset : self.offset_]

        content_encoding = response.get("Content-Encoding")
        if content_encoding is not None:
            if content_encoding == "gzip":
                response = DecompressedResponse(gzip.decompress(response.read()))
            elif content_encoding == "deflate":
                response = DecompressedResponse(zlib.decompress(response.read()))
        if header_length is None:
            content = response.read()
            if verbose:
                print(content)
            try:
                self._result = json.loads(content)
            except UnicodeDecodeError as e:
                raise_error(
                    "Failed to encode using UTF-8. Please use binary_data=True, if"
                    f" you want to pass a byte array. UnicodeError: {e}"
                )
        else:
            header_length = int(header_length)
            content = response.read(length=header_length)
            if verbose:
                print(content)
            self._result = json.loads(content)
            self._output_name_to_buffer_map = {}
            self._buffer = response.read()
            buffer_index = 0
            for output in self._result["outputs"]:
                parameters = output.get("parameters")
                if parameters is not None:
                    this_data_size = parameters.get("binary_data_size")
                    if this_data_size is not None:
                        self._output_name_to_buffer_map[output["name"]] = buffer_index
                        buffer_index = buffer_index + this_data_size

    def as_numpy(self, name: str) -> Optional[np.ndarray]:
        if self._result.get("outputs") is not None:
            for output in self._result["outputs"]:
                if output["name"] == name:
                    datatype = output["datatype"]
                    np_array = None
                    has_binary_data = False
                    parameters = output.get("parameters")
                    dtype = triton_to_np_dtype(datatype)
                    if parameters is not None:
                        this_data_size = parameters.get("binary_data_size")
                        if this_data_size is not None:
                            has_binary_data = True
                            if this_data_size != 0:
                                start_index = self._output_name_to_buffer_map[name]
                                end_index = start_index + this_data_size
                                buffer = self._buffer[start_index:end_index]
                                if datatype == "BYTES":
                                    np_array = deserialize_bytes_tensor(buffer)
                                else:
                                    np_array = np.frombuffer(buffer, dtype=dtype)
                            else:
                                np_array = np.empty(0)
                    if not has_binary_data:
                        np_array = np.array(output["data"], dtype=dtype)
                    assert np_array is not None
                    np_array = np_array.reshape(output["shape"])
                    return np_array
        return None

    def get_output(self, name: str) -> Any:
        for output in self._result["outputs"]:
            if output["name"] == name:
                return output
        return None

    def get_response(self) -> Any:
        return self._result


def parse_inputs(inputs: Dict[str, Any]) -> List[InputData]:
    parsed = []
    for k, v in inputs.items():
        if not isinstance(v, np.ndarray):
            if not isinstance(v, list):
                v = [v]
            if isinstance(v, list):
                elem = v[0]
                if isinstance(elem, str):
                    dtype = np.object
                elif isinstance(elem, bool):
                    dtype = np.bool
                elif isinstance(elem, int):
                    dtype = np.int64
                else:
                    dtype = np.float32
                v = np.array([[vv] for vv in v], dtype)
        parsed.append(InputData(k, v))
    return parsed


class TritonClient:
    def __init__(
        self,
        *,
        url: str = "localhost:2345",
        verbose: bool = False,
        concurrency: int = 1,
        connection_timeout: float = 60.0,
        network_timeout: float = 60.0,
        max_greenlets: Any = None,
        ssl: bool = False,
        ssl_options: Any = None,
        ssl_context_factory: Any = None,
        insecure: bool = False,
    ):
        self.url = url
        # noinspection HttpUrlsUsage
        scheme = "https://" if ssl else "http://"
        self._parsed_url = URL(scheme + url)
        self._base_uri = self._parsed_url.request_uri.rstrip("/")
        self._client_stub = HTTPClient.from_url(
            self._parsed_url,
            concurrency=concurrency,
            connection_timeout=connection_timeout,
            network_timeout=network_timeout,
            ssl_options=ssl_options,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure,
        )
        self._pool = gevent.pool.Pool(max_greenlets)
        self._verbose = verbose

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        *,
        priority: int = 0,
        timeout: Optional[int] = None,
        request_compression_algorithm: Optional[str] = None,
        response_compression_algorithm: Optional[str] = None,
        query_params: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> InferResult:
        inp = parse_inputs(inputs)
        if response_compression_algorithm is None:
            response_compression_algorithm = request_compression_algorithm
        request_body, json_size = get_request(inp, priority, timeout, outputs)
        headers: Dict[str, Any] = {}
        if request_compression_algorithm == "gzip":
            headers["Content-Encoding"] = "gzip"
            request_body = gzip.compress(request_body)
        elif request_compression_algorithm == "deflate":
            headers["Content-Encoding"] = "deflate"
            request_body = zlib.compress(request_body)
        if response_compression_algorithm == "gzip":
            headers["Accept-Encoding"] = "gzip"
        elif response_compression_algorithm == "deflate":
            headers["Accept-Encoding"] = "deflate"
        headers["Inference-Header-Content-Length"] = json_size
        request_uri = f"v2/models/{quote(model_name)}/infer"
        # post
        request_uri = self._base_uri + "/" + request_uri
        if query_params is not None:
            request_uri = request_uri + "?" + _get_query_string(query_params)
        if self._verbose:
            print(f"> POST {request_uri}, headers {headers}\n{request_body}")  # type: ignore
        response = self._client_stub.post(
            request_uri=request_uri,
            body=request_body,
            headers=headers,
        )
        _raise_if_error(response)
        if self._verbose:
            print(response)
        return InferResult(response, self._verbose)

    def classify(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        output_names: List[str],
        *,
        top_k: int = 1,
        priority: int = 0,
        timeout: Optional[int] = None,
        request_compression_algorithm: Optional[str] = None,
        response_compression_algorithm: Optional[str] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> InferResult:
        outputs = [
            {"name": name, "parameters": {"classification": top_k}}
            for name in output_names
        ]
        return self.infer(
            model_name,
            inputs,
            priority=priority,
            timeout=timeout,
            request_compression_algorithm=request_compression_algorithm,
            response_compression_algorithm=response_compression_algorithm,
            query_params=query_params,
            outputs=outputs,
        )

    @staticmethod
    def parse(result: InferResult) -> Dict[str, List[List[Any]]]:
        outputs = result._result.get("outputs")
        if outputs is None:
            return {}
        names = [output["name"] for output in outputs]
        raw: List[np.ndarray] = [result.as_numpy(k) for k in names]
        lists = [arr.tolist() for arr in raw]
        final = []
        for arr_list in lists:
            converted_list = []
            for arr in arr_list:
                converted = [e.decode() if isinstance(e, bytes) else e for e in arr]
                converted = [None if elem == "None" else elem for elem in converted]
                converted_list.append(converted)
            final.append(converted_list)
        return dict(zip(names, final))


class HttpClient:
    _sess: Optional[ClientSession] = None

    def start(self) -> None:
        # TODO : check this
        self._sess = ClientSession(connector=TCPConnector(ssl=False))
    
    async def stop(self) -> None:
        await self._sess.close()
        self._sess = None
    
    @property
    def session(self) -> ClientSession:
        assert self._sess is not None
        return self._sess
