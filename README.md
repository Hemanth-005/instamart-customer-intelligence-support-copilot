# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that (A) turns dirty order exports into a trusted SQLite database, (B) segments customers and predicts who will lapse, (C) triages support tickets with privacy and safety guardrails, and (D) answers policy questions with retrieval and decides refund requests with a tool-calling agent. All data is synthetic (`generate_data.py`); all model calls use the offline `mock_llm.py`; no API key, network or paid service is used. All money is in INR. `generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified.

Layout: `src/data_pipeline.py` (Part A), `src/customer_models.py` (Part B), `src/triage.py` (Part C), `src/policy_rag.py` and `src/refund_agent.py` (Part D), `src/report.py` (SQL, pandas, comparison and demo runs), `run_project.py` (runs everything), `sql/queries.sql`.

## Reproduce

1. `git clone https://github.com/Hemanth-005/instamart-customer-intelligence-support-copilot.git`
2. `cd instamart-customer-intelligence-support-copilot`
3. `python -m venv .venv`
4. `source .venv/bin/activate` (Windows PowerShell: `.venv\Scripts\Activate.ps1`)
5. `pip install -r requirements.txt`
6. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
7. `python -m src.data_pipeline` (Part A; ends with `ALL CHECKPOINTS PASSED`)
8. `python run_project.py` (Parts A to D end to end; prints every number reported below)
9. `python tests_refund_guardrails.py` (optional refund guardrail tests)

`run_project.py` regenerates every file under `data/clean/`, `data/instamart.db`, `outputs/` and `state/`. Seeds are fixed (42) wherever randomness appears.

## Part A

**Audit (raw data).** Blank strings count as missing.

| Table | Shape | Missing values |
|---|---|---|
| customers | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%), others 0 |
| products | 40 x 4 | none |
| orders | 13084 x 9 | delivery_minutes 2541 (19.42%), others 0 |

Data types: customers and orders are read as text except orders `quantity` and `discount_pct` (int); products `price_inr` is int; `order_amount` and `delivery_minutes` are text in the raw file because they contain `N/A`, `error` and blanks.

| Audit figure | Value |
|---|---|
| raw order rows / exact duplicates / pending rows / distinct order_id | 13,084 / 119 / 656 / 12,309 |
| non-numeric order_amount / negative delivery_minutes / blank delivery_minutes | 120 / 106 / 2,541 |
| blank age / ages outside 10-100 / raw city spellings | 61 / 6 / 24 |
| raw gender spellings (non-blank) / blank gender | 10 / 30 |

**Cleaning rules (customers).**
1. City: strip spaces and title-case; 476 values changed; exactly six cities remain (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184).
2. Gender mapped to male / female / unknown; 920 non-canonical spellings standardised, 30 blanks become unknown (female 610, male 560, unknown 30).
3. Ages outside 10-100 set to missing, then all missing ages (61 blank + 6 invalid = 67) filled with the city median via `groupby().transform`; 0 missing, range 18-60.
4. `signup_date` converted to datetime.

**Cleaning rules (orders), in order.**
1. `order_ts` converted to datetime.
2. Exact duplicates removed: 119 rows (13,084 to 12,965).
3. Per `order_id` the record with the latest `order_ts` kept: 656 earlier pending records removed (12,965 to 12,309).
4. `order_amount` made numeric; 112 unparseable values left after steps 2-3 (120 in the raw file) filled as `round(price_inr x quantity x (1 - discount_pct/100), 2)`; 0 missing.
5. 105 negative `delivery_minutes` (106 in the raw file) set to missing; 774 missing delivery times of completed orders (blank plus negative) filled with the median of the order's product category among completed orders; cancelled orders keep a missing delivery time because they were never delivered.

Result: 12,309 rows (11,110 completed, 1,199 cancelled), no pending rows, 0 missing amounts, 0 missing delivery times among completed orders, 1,199 among cancelled; completed revenue INR 2,998,668.80.

**Store and validate.** Clean CSVs are in `data/clean/`; the SQLite database `data/instamart.db` holds tables `customers`, `products`, `orders` (`order_ts` stored as `YYYY-MM-DD HH:MM:SS` text). `validate()` asserts 14 reference checkpoints (city and gender counts, age range, row counts, status counts, no pending rows, amount sum, delivery rules, database tables) and prints `ALL CHECKPOINTS PASSED`.

**SQL results** (`sql/queries.sql`, run with `pd.read_sql_query`):

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; total INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers without a completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 ranking windows | 1,175 rows; top per city: Bengaluru C0082 12,856.20, Chennai C0930 10,151.40, Delhi C0291 13,742.60, Hyderabad C0743 12,730.05, Mumbai C0693 14,766.55, Pune C0426 10,377.10; 8 rows have RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative INR 2,998,668.80 |

Ties make RANK differ from ROW_NUMBER because tied customers share one rank (and the next rank is skipped) while ROW_NUMBER still gives each a distinct sequence number.

**pandas cross-check.** The `pd.pivot_table(..., margins=True)` grand total is INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**Completed order_amount statistics.** Mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 and 645.30; 0 outliers below, 897 above.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**Exploratory answers.** The dataset covers six months of quick-commerce activity (Apr-Sep 2025) in six cities. It contains 11,110 completed orders with a combined value of INR 2.998 million. Order values are concentrated at the lower end, with a long right tail: the mean is 269.91, the median is 135.00 and skewness is 4.61. There are 897 values above the INR 645.30 upper fence. These are not automatically data errors; 71% are electronics orders, which helps explain the high values, and no observations fall below the lower fence. The measured relationships are weak. The largest absolute correlation is between order_amount and discount_pct (-0.095), followed by order_amount and quantity (+0.087). These small correlations do not show that one variable causes another.

## Part B

**RFM** (snapshot 2025-10-01, completed orders): 1,175 customers; recency median 19, min 0, max 180; frequency mean 9.46, max 33; monetary mean 2,552.06, max 14,766.55.

**K-means** (StandardScaler, `n_init=10`, `random_state=42`):

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

I selected **K=4** because it gives the strongest combination of separation and a manageable number of customer groups. From K=3 to K=4, inertia drops by 525, while moving from K=4 to K=5 reduces it by only another 126. K=4 also records the highest silhouette score, 0.526. This gives us four groups that can be interpreted and used for four different campaigns. Inertia naturally decreases as more clusters are added, so choosing a K only because its inertia is lower would not be enough; the elbow pattern, silhouette score and usefulness of the resulting segments all matter.

**Segments** (cluster numbers are only labels with no ordering):

| Segment id | Name | Size | Mean recency | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium (is_premium = True) | 138 | 24.1 | 6.8 | 7,129.7 | Early access to electronics offers, priority support |
| 2 | Loyal Frequent | 419 | 8.8 | 19.4 | 3,410.0 | Subscription / bundle offers, protect with reliable delivery |
| 0 | Occasional | 436 | 28.9 | 3.8 | 1,057.6 | Nudges and category discounts to raise frequency |
| 3 | Lapsing | 182 | 109.0 | 2.2 | 686.0 | Win-back coupon before they churn completely |

PCA (2 components) explained variance ratio: 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium; Premium is flagged for exactly the highest-mean-monetary segment (138 customers).

**Lapse problem.** Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff; 443 lapsed (38.32%). Split `test_size=0.2, stratify=y, random_state=42`: 924 train, 232 test. Features (orders before 2025-09-01 only): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure days. Baseline "never lapsed": accuracy 61.6%, lapsed F1 0.

**Leakage controls.** No order dated 2025-09-01 or later enters any feature (only the label uses September); the segment labels from the RFM clustering are not used as features; imputers, the `StandardScaler` and `OneHotEncoder(drop="first")` live inside a scikit-learn `Pipeline` fitted on training rows only, and the oversampler touches training rows only, so no encoder, scaler or resampler saw test rows. The scaler learned a frequency mean of 8.115 (training rows) versus 8.029 for the full cohort, because the scaler sees only the 924 training rows, not all 1,156. An assertion in the code confirms that no preprocessed feature column contains a missing value (train and test).

**Tuning** (`GridSearchCV`, cv=5, scoring f1):

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| Random forest (100 trees) | max_depth {3,5,8} x min_samples_leaf {5,10,20} | depth 8, leaf 10 | 0.687 |

**Test set, lapsed class:**

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (never lapsed) | - | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[110, 33], [32, 57]] | 0.633 | 0.640 | 0.637 |

**Selected model: random forest.** I chose the random forest using the cross-validation results, rather than using the test set to make the choice. Its best CV F1 is 0.687, compared with 0.651 for logistic regression, and it can model non-linear patterns between purchase frequency and recency. The test results are very close: logistic regression has an F1 of 0.648 and the random forest has 0.637. In this use case, failing to identify a customer who is about to lapse could mean losing their future purchases, while sending an unnecessary coupon to a loyal customer has a smaller cost. That makes recall important. Both models have recall near 0.64, and the oversampling experiment below explores how to increase it. The selected random forest's F1 of 0.637 is above the required 0.55.

**Class-balance experiment** (training lapsed rows duplicated with replacement, `random_state=42`, to 570 vs 570; logistic regression refit; original test set):

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Oversampled | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20) while precision fell slightly (-0.06): the retention team catches many more lapsing customers (76 of 89 versus 58) at the price of more wasted offers (53 false alarms versus 32).

**Interpretation.** Top five random-forest features: monetary (0.280), frequency (0.278), recency (0.109), average delivery minutes (0.060), cancellation rate (0.052): lapse is driven mostly by how much and how often someone buys and how recently. Logistic coefficients: frequency (-1.70, more orders means less lapse risk), favourite category Electronics (-1.14, electronics buyers are stickier than the dropped baseline category) and favourite category Grocery (-0.45, also lower risk than baseline).

**Persistence.** The selected model is saved to `outputs/lapse_model.joblib`; the reloaded model reproduces the test predictions exactly (asserted in code).

**Retention list.** Among test customers in the Premium segment, the top 10 by predicted lapse probability are in `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed); 1 of the 10 actually lapsed. The Premium flag uses whole-window RFM that includes September, so the retention list is illustrative and not a leakage-free evaluation.

## Part C

**Guardrails.** Empty tickets (4) get `skipped_empty`; tickets above 400 estimated tokens (4, each 734 tokens) get `flagged_long`; neither is ever sent to the model.

**Redaction** (email, then phone, then order id, with the specified regexes), over all 164 tickets: 55 emails, 91 phones, 132 order ids, 278 in total; re-applying the patterns to the redacted texts finds 0. The user message is exactly the redacted text, and an assertion wrapping every model call checks that no prompt contains a match.

**Prompts.** Zero-shot: contains CLASSIFY, the four categories and the JSON-only format, no `EXAMPLES:`. Few-shot: the same plus an `EXAMPLES:` line with three labelled examples (fraud, delivery, technical).

**Retries.** Up to 3 retries per ticket, wait = `0.05 x 2**n x uniform(0.9, 1.15)`, one log line per retry; `mock_llm.reset()` before each run. A full run over the 156 eligible tickets raised 19 TransientLLMErrors, all recovered (19 retry lines, every one retry 0, waits between 0.0453 and 0.0562 seconds); no ticket ended in a permanent error.

**Output validation.** Valid JSON with an allowed category gives `classified_json`; otherwise the words from `re.findall(r"[a-z]+", text.lower())` are checked and exactly one allowed word gives `classified_keyword`; everything else (including JSON category "shipping") is `flagged_invalid_output` and goes to human review. One full run: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output (plus 4 skipped_empty, 4 flagged_long).

**Zero-shot versus few-shot** (no budget):

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 / 147) | 7,372 | 50.1 |
| Few-shot | 147 | 100% (147 / 147) | 12,520 | 85.2 |

Zero-shot confusion table (rows true, columns predicted):

| True \ Predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Using few-shot examples increases usage by about 35 tokens per ticket (+70%), but in this run it also eliminates all 21 classification errors. I would accept that extra cost when a mistake has serious consequences. Fraud is the clearest example: 11 of 34 fraud tickets were labelled as billing by the zero-shot prompt, which could cause the 2-hour security SLA to be missed. For a low-risk category where an incorrect label has little impact, the lower-cost zero-shot prompt may be sufficient.

**Budget.** With zero-shot and `TOKEN_BUDGET = 6000`, 29 tickets end `budget_exhausted` and go to human review (`outputs/triage_budget_demo.csv`, 6,009 tokens used).

**Routing** (`outputs/triage_results.csv`, 164 rows, few-shot, `TOKEN_BUDGET = 100000`, reads `outputs/customer_segments.csv`): skipped, long, invalid-output and budget-exhausted tickets go to human_review (priority NORMAL); fraud goes to security_team (HIGH); premium customers go to premium_desk (HIGH); the rest go to support_queue (NORMAL); customers missing from the segment file count as not premium. Result: support_queue 102, security_team 34, human_review 17 (9 invalid output, 4 empty, 4 long), premium_desk 11; HIGH priority 45. Rule-validated (JSON or keyword check passed): 147 tickets are routed automatically; the 17 human_review tickets were not trusted to the model.

## Part D

**Chunking.** 40-word chunks with 10-word overlap give 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2.

**Retrieval.** The `TfidfVectorizer` first learns word weights from the policy chunks. The same vectorizer then converts a user's question into that feature space, and cosine similarity is used to rank the chunks; `retrieve(query, k=3)` returns the source, score and text for the top results. Term frequency (TF) gives weight to words that occur in a chunk, while inverse document frequency (IDF) reduces the influence of words that appear across many chunks. As a result, uncommon and more informative terms can have greater influence on the match. This method compares wording rather than understanding meaning, so a differently phrased question such as "money back" may not match a chunk that only says "refund" very well. For the query "refund policy late delivery", the top three retrieved chunks all come from `refund_policy.txt`.

**Answering.** `answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message with POLICY ASSISTANT (answer only from context, otherwise say the information could not be found) and a user message `CONTEXT:` / `[source]` + chunk text / `QUESTION:` / question; it returns the text and sources.

**Evaluation** (`policy_eval.csv`, 14 questions):

| Setting | Chunks | Expected doc in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Fallback (2 unanswerable) |
|---|---|---|---|---|
| Overlap 10 | 15 | 12 / 12 | 12 / 12 | 2 / 2 |
| Overlap 0 | 13 | 12 / 12 | 10 / 12 | 2 / 2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" and "How soon is money returned when a payment fails but is deducted?". The sentence holding the answer is cut across a chunk boundary, so the retrieved chunk contains only part of it and the expected fact is missing; the 10-word overlap keeps such sentences whole in at least one chunk.

**Tools.** `TOOLS_SPEC` (name, description, JSON parameter spec) and `TOOL_MAP` define `get_order` (reads `data/instamart.db`), `search_policy` (returns `results`), `calculate_refund` (over 45 minutes refunds 100%, otherwise 0; the completed-only rule is enforced by the agent flow and by the checks inside `issue_refund` and `escalate_to_human`), `issue_refund` and `escalate_to_human` (each appends a timestamped JSON line to `outputs/refund_ledger.jsonl`). Refunds above INR 1000 are escalated, never issued. `issue_refund` and `escalate_to_human` also validate the request independently before writing to the ledger: the order must exist, its status must be completed, delivery must have taken over 45 minutes, and the requested amount must match the policy amount. `issue_refund` accepts amounts up to INR 1000, while amounts above INR 1000 are sent for escalation. Invalid requests return an error and do not create a ledger entry. Run `python tests_refund_guardrails.py` to check five invalid attempts.

**ReAct loop** (`run_agent`, limits: 6 model calls, 6000 tokens, 10 seconds). Seven scenario results (`outputs/agent_runs.jsonl`, one line per run with every step):

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 897 |
| S2 | REFUND | 135.00 | 5 | 897 |
| S3 | ESCALATE | 1999.00 | 5 | 884 |
| S4 | DENIED | 0.00 | 4 | 605 |
| S5 | DENIED | 0.00 | 3 | 337 |
| S6 | NOT_FOUND | 0.00 | 2 | 72 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3000 |

All refund amounts equal `expected_refund_inr`; S7 uses all six model calls under the default token limit and falls back to a human.

**Stop-safely demonstrations.**
- S1 with `token_limit=200`: status `FALLBACK_TOKEN_LIMIT` (stops right after the over-limit model call, without executing the requested tool).
- S1 with `time_limit_seconds=0`: status `FALLBACK_TIMEOUT` (stops before the first model call).

Each fallback returns the steps so far and a short human-handoff message in `final_answer`.

**Crash and resume (S1).** A checkpoint is written to `state/<run_id>.json` after every tool observation (run id, scenario id, full messages, tokens, step number, tools executed). A simulated crash is raised right after the second tool and `resume_agent(run_id)` continues. Tools executed across both parts, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated); final status REFUND, INR 128.25.

**ReAct versus plan-and-execute.** In my implementation, ReAct used 5 model calls for the approved refund (S1), 4 for the cancelled order (S4), 3 for the on-time order (S5) and 2 for the unknown order (S6). Their token counts were 897, 605, 337 and 72 respectively. Each ReAct call includes the conversation history built up so far, which explains why the token use can grow across steps. A plan-and-execute agent could first request a complete sequence of actions and might complete S1 in about 2 model calls with fewer tokens. Its weakness is that the plan is made before the tool results are known. If an order is NOT_FOUND or is cancelled, the agent must stop the remaining actions and adjust the plan, potentially adding calls or wasting planned work. ReAct instead checks each observation before choosing the next action: it finished the NOT_FOUND case in 2 calls and rejected the cancelled order before running a money-related tool. For refunds, I prefer ReAct because order status, delivery time and the INR 1000 threshold must be checked as the workflow progresses. The iteration, token and time limits, together with checkpoints, help control the extra calls.

## Limitations and Next Steps

- The retention list is illustrative and not a leakage-free evaluation: the Premium flag comes from whole-window RFM that includes the September window defining the lapse label (a post-hoc label). The model's own features use only orders before 2025-09-01.
- Test set is small (232 rows, 89 lapsed): F1 differences of about 0.01 between models are noise. Next: repeated cross-validation, probability calibration, a time-based split.
- The mock LLM is rule-based; accuracy numbers (85.7% / 100%) describe the mock, not a real model. Next: evaluate a real model on a labelled sample behind the same guardrails.
- Retrieval is TF-IDF only and breaks on paraphrases; next: semantic embeddings and a re-ranker.
- Refunds are a local JSONL ledger, not a payment system; next: idempotency keys so a resumed run can never pay twice, access control and an audit trail.
- The agent executes the first tool call only; parallel tool calls are not supported.
