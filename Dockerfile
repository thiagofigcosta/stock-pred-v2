FROM python:3.8 AS builder

COPY requirements.txt .

RUN pip install --user -r requirements.txt

FROM python:3.8-slim
WORKDIR /code

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz r-base make gcc libc6-dev 

# install irace under /usr/local/lib/R/site-library/irace
RUN echo "install.packages('irace', repos='http://cran.us.r-project.org')" | R --no-save
  
COPY --from=builder /root/.local /root/.local
COPY entrypoint.sh .
COPY ./*.py ./
COPY ./irace ./irace/
COPY ./datasets ./datasets/

RUN chmod +x entrypoint.sh

ENV PATH=/root/.local/bin:$PATH

CMD ["./entrypoint.sh"]