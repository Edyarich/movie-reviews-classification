FROM python:3.10

WORKDIR /app

COPY requirements.txt light_model.keras inference.py /app/

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "inference.py", "light_model.keras" ]
