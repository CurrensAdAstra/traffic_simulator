FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# SUMO: Ubuntu apt 버전(1.12)은 너무 오래되어 최신 net.xml의 vClass(drone/cable_car 등)를
# 인식하지 못함 → 공식 PPA(sumo/stable)에서 최신 안정판 설치.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       python3-dev \
       ca-certificates \
       software-properties-common \
       gnupg \
    && add-apt-repository -y ppa:sumo/stable \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       sumo \
       sumo-tools \
    && rm -rf /var/lib/apt/lists/*

ENV SUMO_HOME=/usr/share/sumo

# CUDA 후처리 + 통계 계산 의존성
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
       numpy \
       cupy-cuda12x

WORKDIR /workspace

# 기본 명령: 도구 버전 진단
CMD ["bash", "-lc", "python3 --version && sumo --version | head -n 1"]
