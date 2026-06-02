.PHONY: install validate train register promote serve mlflow-ui predict monitor docker-build docker-run clean

PORT    ?= 8000
PYTHON  ?= MlFlowStarter/venv/bin/python
UVICORN ?= MlFlowStarter/venv/bin/uvicorn
MLFLOW  ?= MlFlowStarter/venv/bin/mlflow

install:
	$(PYTHON) -m pip install -r requirements.txt

validate:
	$(PYTHON) -c "\
from data.loader import load; \
from data.validator import validate; \
d = load(); \
w = validate(d, log_to_mlflow=False); \
print('Warnings:', w if w else 'none')"

train:
	$(PYTHON) -m ml.train

register:
	$(PYTHON) -m ml.register register

promote:
	$(PYTHON) -m ml.register promote

serve:
	$(UVICORN) serve.app:app --host 0.0.0.0 --port $(PORT) --reload

mlflow-ui:
	$(MLFLOW) ui --port 5000

predict:
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H 'Content-Type: application/json' \
		-d '{"instances": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]}' \
		| $(PYTHON) -m json.tool

monitor-drift:
	$(PYTHON) -m monitor.drift

monitor-log:
	$(PYTHON) -m monitor.logger

docker-build:
	docker build -t mlops-playground:latest .

docker-run:
	docker run --rm -p $(PORT):8000 mlops-playground:latest

clean:
	rm -rf mlruns models data_cache __pycache__ */__pycache__
