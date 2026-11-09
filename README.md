# Customer Intelligence & Support Copilot — Teaching Kit

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run

Python 3.11+ recommended.

### 1. Clone the repository

```bash
git clone https://github.com/Hemanth-005/instamart-customer-intelligence-support-copilot.git
cd instamart-customer-intelligence-support-copilot
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

```bash
python generate_data.py
```

### 5. Run validation

```bash
python -c "from src.data_pipeline import validate; validate()"
```

Expected output:

```text
ALL CHECKPOINTS PASSED
```

### 6. Run the project

```bash
python run_project.py
```

You can also open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` in VS Code or Jupyter and run all cells from beginning to end.

All model calls use the supplied offline mock implementation. No paid API key is required.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

## Part B — Segments and lapse
RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes. The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV and a saved joblib model. The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

## Part C — Ticket triage and privacy
Empty and overlong tickets are skipped before model calls. Email, phone and order ID patterns are redacted before prompting. Retries handle simulated transient errors. JSON category values are allow-listed; malformed outputs are accepted only when exactly one allowed category is detectable using `re.findall`. Human review receives guardrail/invalid/budget cases, fraud goes to security, premium customers go to the premium desk, and the rest go to the support queue.

## Part D — Policy retrieval and refund tools
Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences. Refund rules are implemented in Python tools backed by SQLite: completed orders delivered in over 45 minutes qualify for a full refund; orders over INR 1000 escalate instead of auto-issuing. This is a teaching simulation, not a payment integration.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- The retention list is illustrative and not a leakage-free evaluation because the Premium flag is derived from whole-window RFM, which includes the September lapse-label period; the prediction model itself uses only pre-cutoff features.
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.
