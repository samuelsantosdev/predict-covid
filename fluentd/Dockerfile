FROM fluent/fluentd
RUN fluent-gem install fluent-plugin-elasticsearch
COPY ./tmp/fluentd.conf /fluentd/etc 

ENV FLUENTD_CONF=fluentd.conf
ENV ELASTICSEARCH_HOST=elasticsearch
ENV ELASTICSEARCH_PORT=9200
ENV ELASTICSEARCH_USER=elastic
ENV ELASTICSEARCH_PASS=qwe123