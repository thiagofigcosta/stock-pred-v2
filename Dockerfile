FROM python:3.8 AS builder

COPY requirements.txt .

RUN pip install --user -r requirements.txt

FROM python:3.8-slim
WORKDIR /code

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz

COPY --from=builder /root/.local /root/.local
COPY entrypoint.sh .
COPY ./*.py ./

RUN chmod +x entrypoint.sh

ENV PATH=/root/.local/bin:$PATH

CMD ["./entrypoint.sh"]