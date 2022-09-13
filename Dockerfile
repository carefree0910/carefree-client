FROM python:3.8-slim-buster AS builder
WORKDIR /carefree-client
COPY . .

RUN apt-get clean && \
    apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv .venv &&  \
    .venv/bin/pip install -U pip setuptools && \
    .venv/bin/pip install . --default-timeout=10000 && \
    find /carefree-client/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM python:3.8-slim
WORKDIR /carefree-client
COPY --from=builder /carefree-client /carefree-client
ENV PATH="/carefree-client/.venv/bin:$PATH"
COPY apis apis

EXPOSE 8123
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "8123"]