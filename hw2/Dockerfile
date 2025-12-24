FROM python:3.9-slim

WORKDIR /app

COPY src/requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install build

COPY . .

CMD ["bash"]