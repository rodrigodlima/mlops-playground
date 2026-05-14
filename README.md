# mlops-playground

repo/
├── prompts/
│   └── plan_explainer.txt
├── evals/
│   ├── dataset.jsonl        # input + expected behavior
│   └── run_eval.py          # run prompt, compare with expected
└── .github/workflows/
    └── eval.yml             # trigger PR