# Jupyter Notebooks — Quick Explainer

## What it is

A Jupyter Notebook (`.ipynb`) is an interactive document that mixes **executable code**, **rich text (Markdown)**, **equations (LaTeX)**, and **visualizations** in a single file. Originally part of the IPython project, "Jupyter" stands for **Ju**lia + **Pyt**hon + **R**, the three core languages it was built to support.

It runs in the browser (or in IDEs like VS Code) and is the de-facto tool for data science, ML experimentation, prototyping, and teaching.

## How it works

A notebook has three main pieces:

1. **Notebook file (`.ipynb`)** — a JSON file storing cells, outputs, and metadata.
2. **Kernel** — a language-specific process (e.g. `ipykernel` for Python) that actually runs the code. The notebook UI sends code to the kernel and renders whatever comes back.
3. **Frontend** — JupyterLab, classic Notebook, VS Code, or any client that speaks the Jupyter protocol over ZeroMQ/WebSockets.

When you run a cell, the frontend ships the source to the kernel, the kernel executes it in a persistent in-memory session, and the result (stdout, return value, plots, errors) is streamed back and stored inline in the notebook file.

## Cell types

- **Code cells** — executed by the kernel. Each cell shares state with prior cells (variables, imports, functions persist across cells).
- **Markdown cells** — rendered as formatted text. Supports headings, lists, links, images, LaTeX math (`$...$`), and code fences.
- **Raw cells** — passed through unmodified (rarely used; for export pipelines).

Each code cell has an **execution counter** (`In [n]:`) that increments per run — a quick way to see execution order, which is *not* required to match cell order.

## Execution model — the gotcha

Cells can be run **in any order**, and the kernel keeps state across runs. That means:

- Re-running a cell after editing earlier cells can give stale results.
- Deleting a cell doesn't remove the variables it defined from kernel memory.
- A notebook that looks correct top-to-bottom may not actually run that way from a fresh kernel.

**Rule of thumb:** before sharing or committing, run **Kernel → Restart & Run All** to confirm the notebook works linearly from a clean state.

## Common commands

| Shortcut | Action |
|---|---|
| `Shift + Enter` | Run cell, move to next |
| `Ctrl/Cmd + Enter` | Run cell, stay |
| `Alt + Enter` | Run cell, insert new below |
| `Esc` then `A` / `B` | Insert cell above / below |
| `Esc` then `M` / `Y` | Convert to Markdown / Code |
| `Esc` then `D D` | Delete cell |
| `Esc` then `0 0` | Restart kernel |

## Setup (this project)

```bash
python -m venv venv
source venv/bin/activate
pip install jupyter ipykernel
jupyter notebook        # or: jupyter lab
```

Open `basic.ipynb` in the browser, or open it directly in VS Code with the Jupyter extension.

## Strengths

- Fast iteration loop — edit, run, inspect, repeat.
- Inline plots and rich output (Pandas tables, images, HTML, widgets).
- Self-documenting via Markdown cells next to the code.
- Great for exploration, EDA, demos, and reproducible analysis.

## Weaknesses

- Hidden state from out-of-order execution makes bugs sneaky.
- `.ipynb` is JSON with embedded outputs → ugly diffs in git. Tools like `nbstripout` or `jupytext` help.
- Not ideal for production code — refactor stable logic into `.py` modules and import them into the notebook.
- Testing is awkward compared to plain Python.

## When to use vs. `.py`

| Use notebook for | Use `.py` for |
|---|---|
| EDA, prototyping, one-off analysis | Reusable libraries, modules |
| Teaching, demos, reports | Production code, services |
| Visual exploration of data | Code that needs unit tests / CI |
| Sharing results with narrative | Anything imported elsewhere |

A common workflow: prototype in a notebook, then **promote** stable code into `.py` modules and import them back.
