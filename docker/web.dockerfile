FROM python:3.10-slim

WORKDIR /app
COPY webapp.py requirements.txt /app/
COPY index.html /app/templates/

RUN pip install --no-cache-dir -r requirements.txt

ENV GATEWAY_URL=http://sentiment-gateway/predict
EXPOSE 8080
ENTRYPOINT ["python","webapp.py"]
