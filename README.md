# stock-pred-v2

based on: https://github.com/thiagofigcosta/Stock-Pred-LSTM


docker build -t stock-pred:2.0.0 .

docker run -d stock-pred:2.0.0

docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ') bash

docker stop $(docker container ls | grep stock-pred | cut -f 1 -d' ')
