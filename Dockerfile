FROM tensorflow/tensorflow:1.15.2

WORKDIR /home

COPY . /home

RUN apt-get -y update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    pip install --upgrade pip

RUN pip install gym_super_mario_bros && \
    pip install opencv-python
