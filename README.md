# polarization-project

This project measures political polarization in online discussions using LLM-based pipelines on signed social networks. It uses the Reddit Politosphere dataset to compare interaction-level and graph-structural polarization measures.

## Setup

Create and activate a virtual environment, then install the project and its dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Install and start [Ollama](https://ollama.com/), then pull the model configured in `config/config.yaml`:

```bash
ollama pull llama3.2
ollama serve
```

Update `llm_model_name` and `llm_base_url` in the configuration as needed.

## Tests

Run the test suite with:

```bash
pytest
```

## Ownership

- Yasmina owns `src/dataset/`, `experiments/phase2_data_pipeline_check/`, and `experiments/rq2_condition_breakdown/`.
- Kareem owns `src/algorithms/`, `experiments/phase1_synthetic_validation/`, and `experiments/rq1_pipeline_comparison/`.
- `src/common/` is shared and jointly owned.
