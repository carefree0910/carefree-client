`carefree-client` is a client framework based on `FastAPI`. It also supports interacting with Triton Inference Server.


## Build

```bash
docker build -t cfclient .
```

If your internet environment lands in China, it might be faster to build with `Dockerfile.cn`:

```bash
docker build -t cfclient -f Dockerfile.cn .
```


## Run

```bash
docker run --rm -p 8123:8123 -v /full/path/to/your/client/logs:/carefree-client/apis/logs cfclient:latest
```

or

```bash
docker run --rm --link image_name_of_your_triton_server -p 8123:8123 -v /full/path/to/your/client/logs:/carefree-client/apis/logs cfclient:latest
```

In this case, you need to modify the `apis/interface.py` file as well: you need to modify the `constants` variable (defined at L27) and set the value of `triton_host` (defined at L28) from `None` to `image_name_of_your_triton_server`.
