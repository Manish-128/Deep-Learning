FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean

WORKDIR /model

COPY . /model

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "happy_sad_python.py"]
