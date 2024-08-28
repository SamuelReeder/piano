FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace

COPY . .

RUN pip install -r requirements.txt