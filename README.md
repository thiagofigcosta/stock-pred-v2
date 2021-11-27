# stock-pred-v2

based on: https://github.com/thiagofigcosta/Stock-Pred-LSTM

## Running on Host

### Virtual environment:

To set up virtual env
```
pip3 install virtualenv
python3 -m virtualenv -p python3 venv
```

To activate virtual env
```
source venv/bin/activate
```

Install dependencies and run the application on the activated enviroment!

To deactivate virtual env
```
deactivate
```

To delete venv:
```
rm -rf venv
```

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

To find where is the irace installed on OS:
```
R
> library("irace")
> system.file(package = "irace")
```

You might want to use the full instalation path on the OS.
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

### Running predef hyperparameters on host
```
python3 main.py --train --eval --plot --plot-eval --plot-dataset --save-plots --start-date 01/01/2016 --end-date 07/05/2021 --test-date 07/01/2021 --enrich-dataset --analyze-metrics --use-all-hyper-on-all-stocks --do-not-test-hyperparams-with-more-fields --stock T --stock GOOG
```
You might want to use --force-train
## Running with Docker

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
```

or

```
docker run -e RUN_IRACE_NAS='True' -d stock-pred:2.0.0
```

## Running with Docker-Compose

### To build

```
docker-compose build
```

### To Run

```
docker-compose up -d
```

## Useful Docker commands

### To follow logs

```
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
docker exec -it $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1) bash -c "tar -zcvf /code/exp.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt"
docker cp $(docker container ls | grep stock-pred | cut -f 1 -d' ' | head -n 1):/code/exp.tar.gz .
```

### To uncompress experiments

```
mkdir -p experiment ; tar -zxvf exp.tar.gz -C experiment
```
