.PHONY: install train serve mlflow-ui predict docker-build docker-run clean

PORT ?= 8000

install:
	python -m pip install -r requirements.txt

train:
	python -m ml.train

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port $(PORT) --reload

mlflow-ui:
	mlflow ui --port 5000

predict:
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H 'Content-Type: application/json' \
		-d '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}' | python -m json.tool

docker-build:
	docker build -t mlops-playground:latest .

docker-run:
	docker run --rm -p $(PORT):8000 mlops-playground:latest

clean:
	rm -rf mlruns models __pycache__ */__pycache__
