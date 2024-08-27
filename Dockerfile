FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace

COPY . .

RUN pip install -r requirement.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
