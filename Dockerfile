FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "pbrc.experiments.run_all", "--out", "results/sim", "--seed", "0"]
