FROM python:3.10-slim

WORKDIR /app
COPY proto.py gateway.py sentiment/1/metadata.pkl requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python", "gateway.py", "--metadata", "metadata.pkl"]
