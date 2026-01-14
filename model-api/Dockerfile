FROM python:3.9.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run.py"]