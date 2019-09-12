#!/bin/bash

docker-compose build
docker-compose up -d

while ! docker-compose exec runs-db psql --username=postgres -c 'SELECT 1'; do
  echo 'Waiting for postgres...'
  sleep 1;
done;

docker-compose exec runs python manage.py test
docker-compose exec runs flake8 project
docker-compose exec -e CI=true client npm test

docker-compose down