FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    ultralytics==8.3.* rasterio==1.4.* pyproj shapely affine \
    opencv-python-headless laspy[lazrs] scipy numpy
WORKDIR /app
COPY solution.py /app/solution.py
COPY hillshade_lib.py /app/hillshade_lib.py
COPY best.pt /app/best.pt
COPY best_e2.pt /app/best_e2.pt
ENTRYPOINT ["python", "/app/solution.py"]
