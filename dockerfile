# 基礎映像
FROM python:3.10 as base

# 設置環境變量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED=1

# 安裝libGL依賴
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_with_edge_torch.txt.txt /requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /requirements.txt -U

# 一次性安裝所需的 Python 套件
# RUN pip install --upgrade pip && \
#     pip install -r requirement.txt


# 設置工作目錄
WORKDIR /app

# 將應用程序代碼拷貝到容器中
# COPY . /app

# 設置容器啟動時的默認命令為 /bin/sh
CMD ["/bin/sh"]
