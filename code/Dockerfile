FROM nvidia/cuda:12.0.0-devel-ubuntu20.04 as builder
ARG DEBIAN_FRONTEND=noninteractive

COPY ./server /app/server
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh

RUN apt update

# Install packages for app and core
RUN apt-get update && apt-get upgrade -y
RUN apt install --no-install-recommends -y libncursesw5-dev tk-dev \
    libc6-dev libxml2-dev libxslt1-dev checkinstall python-dev git curl \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# # Install python
RUN apt-get update && apt-get install -y python3.8 python3-pip && \
    pip3 install -U pip

# comment out the following line if you face an error related to blinker
WORKDIR /usr/lib/python3/dist-packages
RUN rm -rf blinker* *yaml* PyYAML-5.3.1.egg-info
WORKDIR /app/server

RUN pip3 install -r requirements.txt

# # Install ecos_core
RUN mkdir -p /app/server/tmp/ \
    && curl "https://ecos-ai-test-upload.s3.ap-northeast-1.amazonaws.com/ecos_core-1.3.1.zip" -o "/app/server/tmp/ecos_core.zip" \
    && pip3 install --default-timeout=100 "/app/server/tmp/ecos_core.zip" \
    && rm -r /app/server/tmp/

# Second stage
FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.8 python3-pip && \
    pip3 install -U pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages

ENV PYTHONUNBUFFERED 1

WORKDIR /app/server