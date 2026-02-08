# Temporal vLLM Latency POC

This POC uses Temporal to orchestrate a mocked workflow that simulates vLLM latency incidents, diagnoses likely causes, applies configuration fixes, and validates improvement. All telemetry/logs are mocked to demonstrate the end-to-end flow.

## Prereqs

- Python 3.12+
- Temporalite (local Temporal server)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Temporalite

```bash
temporalite start --namespace default
```

## Run the workflow

```bash
python src/runner.py --scenario default
```

Other scenarios:

```bash
python src/runner.py --scenario oom_pressure
python src/runner.py --scenario tokenizer_cpu_bottleneck
```

## Output

- Console summary of the simulated diagnosis and improvements
- JSON report written to `./reports/latency_report.json`

## GUI

Backend API (serves static assets if built):

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn gui.api:app --port 8080
```

Frontend (React + Vite):

```bash
cd gui
npm install
npm run dev
```

Build + deploy (single server):

```bash
cd gui
npm install
npm run build
cd ..
PYTHONPATH=src uvicorn gui.api:app --port 8080
```

## Notes

- All telemetry is mocked from `src/mock_data/` and perturbed per scenario.
- The workflow is designed to show how Temporal orchestrates each step and passes artifacts between activities.
