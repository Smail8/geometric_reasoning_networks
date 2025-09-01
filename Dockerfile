FROM nvidia/cuda:12.9.0-runtime-ubuntu20.04

RUN apt-get update  -y
RUN apt-get upgrade  -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update  -y
RUN apt install python3.8 -y
RUN apt install python3-pip -y

WORKDIR /GRN
COPY . /GRN
RUN pip install --upgrade pip
RUN apt-get update && apt-get install libsm6 libxext6 -y
RUN apt-get install libgl1-mesa-dev -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN apt-get install python3-opencv -y
RUN pip3 install -r requirements.txt

EXPOSE 8080