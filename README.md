# mlops-playground

```
repo/
├── ml/
│   ├── train.py             # train iris RF, log to MLflow, persist joblib
│   └── predict.py           # load model, predict (CLI + library)
├── serve/
│   └── app.py               # FastAPI: /health, /predict
├── models/                  # generated; trained model artifact (gitignored)
├── mlruns/                  # generated; MLflow tracking store (gitignored)
├── prompts/
│   └── plan_explainer.txt
├── evals/
│   ├── dataset.jsonl        # input + expected behavior
│   └── run_eval.py          # run prompt, compare with expected
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Classical ML POC (train + serve)

End-to-end iris classifier: scikit-learn training, MLflow tracking, FastAPI serving, Docker packaging.

### Local

```bash
make install       # pip install -r requirements.txt
make train         # trains, writes models/iris_rf.joblib, logs run to mlruns/
make serve         # uvicorn on :8000
make predict       # curl /predict in another shell
make mlflow-ui     # browse runs at http://localhost:5000
```

### Docker

```bash
make docker-build  # bakes a trained model into the image
make docker-run    # serves on :8000
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```
