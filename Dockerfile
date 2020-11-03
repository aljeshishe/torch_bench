# FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3
FROM nvcr.io/nvidia/pytorch:20.10-py3

# Build Arguments
#ARG PREFECT_VERSION
#ARG EXTRAS=kubernetes
#ARG GIT_SHA
#ARG BUILD_DATE

# Set system locale
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

## Image Labels
#LABEL maintainer="help@prefect.io"
#LABEL io.prefect.python-version=${PYTHON_VERSION}
#LABEL org.label-schema.schema-version = "1.0"
#LABEL org.label-schema.name="prefect"
#LABEL org.label-schema.url="https://www.prefect.io/"
#LABEL org.label-schema.version=${PREFECT_VERSION}
#LABEL org.label-schema.vcs-ref=${GIT_SHA}
#LABEL org.label-schema.build-date=${BUILD_DATE}
# COPY config /root/.ssh/config

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python main.py