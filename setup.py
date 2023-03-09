from setuptools import setup, find_packages

VERSION = "0.1.9"
DESCRIPTION = "Client framework based on FastAPI"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-client",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pillow",
        "numpy>=1.22.3",
        "email-validator",
        "ftfy",
        "regex",
        "dill",
        "tqdm",
        "sqlmodel",
        "PyYaml",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "requests",
        "starlette",
        "python-multipart",
        "requests_toolbelt",
        "gevent",
        "geventhttpclient",
        "aiohttp",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python carefree-learn PyTorch",
)
