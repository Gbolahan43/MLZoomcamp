FROM python:3.11.5-slim

RUN pip install pipenv

WORKDIR /ml-zoomcamp

COPY ["dv.bin", "cr_rfmodel.bin", "scaler.bin", "predict.py", "./"]
COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
