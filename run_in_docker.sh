#!/bin/bash
set -ex

docker build . -t torch_bench
docker run -it torch_bench 
