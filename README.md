# stock-pred-v2

based on: https://github.com/thiagofigcosta/Stock-Pred-LSTM

## Host

### Install deps linux:
Install dependencies:
```
pip3 install -r requirements.txt
```

Install graphviz
```
sudo yum install graphviz
```
or
```
sudo apt install graphviz
```
or
```
sudo zypper install graphviz
```


Install irace
```
sudo yum install r-base
```
or
```
sudo apt install r-base
```
or
```
sudo zypper install R-base
```

then
```
echo "install.packages('irace', repos='http://cran.us.r-project.org')" | R --no-save
```

### Run irace

```
cd irace
irace
```

### Install deps windows:
Open anaconda console and type:
```
pip install pydot
pip install numpy
pip install keras
pip install --user tensorflow
```
Install [graphviz](https://graphviz.gitlab.io/download/).

## Docker

### To build image

You may want to edit the `entrypoint.sh` to add the commands for your expriment.

```
docker build -t stock-pred:2.0.0 .
```

### To run image deatached

```
docker run -d stock-pred:2.0.0
```

or

```
docker run -e RUN_DEFAULT_EXP='True' -d stock-pred:2.0.0
docker logs --follow $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To access running container

```
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash
```

### To stop running container

```
docker stop $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1)
```

### To copy experiment results compressed

```
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash -c "tar -zcvf /code/exp.tar.gz /code/datasets /code/saved_models /code/saved_plots log.txt"
docker cp $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1):/code/exp.tar.gz .
```

#### To uncompress

```
mkdir -p experiment ; tar -zxvf exp.tar.gz -C experiment
```
