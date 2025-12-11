FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirement.txt

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
