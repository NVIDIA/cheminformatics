#!/bin/bash

BASE_DIR=$(dirname $0)

docker stop cuchem_db
docker rm cuchem_db

docker build ${BASE_DIR} -t cuchem_db_img
docker run --name cuchem_db \
  -p 3306:3306 \
  -d pef_db_img

# Production mysql db command
# docker run \
#   --network host \
#   --name pef_db\
#   -e MYSQL_ROOT_PASSWORD="pef@@12345678" \
#   -p 3306:3306 \
#   -v /raid/benchmarking/db/conf/:/etc/mysql \
#   -v /raid/benchmarking/db/data/:/var/lib/mysql \
#   -v /raid/benchmarking/db/mysql-files/:/var/lib/mysql-files/ \
#   -d mysql:latest
