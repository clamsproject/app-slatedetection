FROM jjanzic/docker-python3-opencv

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./ /app
WORKDIR /app

CMD ["python", "app.py"]