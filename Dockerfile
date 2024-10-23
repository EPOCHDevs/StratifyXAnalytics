# Use the official Python image from the Docker Hub
FROM python:3.10-slim AS stratifyx-analytics

# Set the working directory in the container
WORKDIR /app

RUN apt-get update  \
    && apt-get install -y git  \
    && rm -rf /var/lib/apt/lists/*  \
    && git clone https://github.com/EPOCHDevs/StratifyXAnalytics.git  \
    && cd /app/StratifyXAnalytics  \
    && pip install --no-cache-dir -r requirements.txt \

EXPOSE 9006
CMD ["uvicorn", "start:app", "--host", "0.0.0.0", "--port", "9006"]
