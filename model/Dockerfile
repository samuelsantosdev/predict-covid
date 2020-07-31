FROM ubuntu:18.04
LABEL maintainer="samuelsantosdev@gmail.com"

RUN apt update && apt install -y python3 python3-pip cron \
    gcc python3-dev musl-dev zip && python3 -m pip install --upgrade pip \
    && mkdir /var/task

ADD . /var/task
WORKDIR /var/task

RUN python3 -m pip  install -r requirements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]