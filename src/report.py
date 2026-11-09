"""Report helpers: SQL results, pandas cross-check/EDA (Part A), triage comparison (Part C), policy eval and agent demos (Part D)."""
import json, re, sqlite3, shutil
from pathlib import Path
import numpy as np, pandas as pd
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/"data/instamart.db"; OUT=ROOT/"outputs"; STATE=ROOT/"state"

def sql_report():
    con=sqlite3.connect(DB); text=(ROOT/"sql/queries.sql").read_text()
    qs=[q for q in re.split(r"-- Q\d.*\n",text) if q.strip()]
    print("\n--- SQL results ---")
    for i,q in enumerate(qs,1):
        d=pd.read_sql_query(q,con); info=""
        if i==2: info=f"revenue sum={d.revenue.sum():.2f}; largest={d.sort_values('revenue').iloc[-1].tolist()}"
        if i==5: info=f"average spend={d.avg_spend.iloc[0]:.2f}"
        if i==6: info=f"top per city={d[d.row_num==1][['city','customer_id','total_spend']].round(2).values.tolist()}; rank!=row_number rows={int((d.spend_rank!=d.row_num).sum())}"
        if i==7: info=f"last cumulative={d.cumulative_revenue.iloc[-1]:.2f}"
        print(f"Q{i}: rows={len(d)} {info}")

def eda_report():
    con=sqlite3.connect(DB)
    d=pd.read_sql_query("select o.*,c.city from orders o join customers c using(customer_id) where o.status='completed'",con)
    pv=pd.pivot_table(d.merge(pd.read_sql_query("select product_id,category from products",con),on="product_id"),
        values="order_amount",index="city",columns="category",aggfunc="sum",margins=True)
    print("\n--- pandas cross-check ---"); print("pivot grand total:",round(pv.loc["All","All"],2))
    g=d.groupby("city").agg(total=("order_amount","sum"),mean=("order_amount","mean"),count=("order_amount","count")).round(2); print(g)
    x=d.order_amount; q1,q3=x.quantile(.25),x.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
    print(f"mean={x.mean():.2f} median={x.median():.2f} skew={x.skew():.2f} Q1={q1:.2f} Q3={q3:.2f} fences=({lo:.2f},{hi:.2f}) below={int((x<lo).sum())} above={int((x>hi).sum())}")
    cm=d[["order_amount","discount_pct","quantity","delivery_minutes"]].astype(float).corr().round(3); print(cm)

def part_c_report():
    from src.triage import run_pipeline, ZERO, FEW, PATTERNS
    t=pd.read_csv(ROOT/"data/raw/tickets.csv",keep_default_na=False)
    orig=mock_llm.chat; st={"transient":0,"calls":0}
    def guarded(messages,tools=None):
        for m in messages[1:]:
            assert not any(p.search(m["content"]) for _,p,_ in PATTERNS),"PII leaked into a prompt"
        st["calls"]+=1
        try: return orig(messages,tools)
        except mock_llm.TransientLLMError: st["transient"]+=1; raise
    mock_llm.chat=guarded
    res={}
    try:
        print("\n--- Part C ---")
        import io, contextlib
        for name,p in [("zero-shot",ZERO),("few-shot",FEW)]:
            st["transient"]=0
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                r,s=run_pipeline(prompt=p,token_budget=10**9)
            waits=[float(m.group(1)) for m in re.finditer(r"wait=([\d.]+)s",buf.getvalue())]
            cl=r[r.status.isin(["classified_json","classified_keyword"])].copy()
            cl["true"]=t.set_index("ticket_id").loc[cl.ticket_id,"true_category"].values
            acc=(cl.category==cl.true).mean()
            print(f"{name}: classified={len(cl)} correct={(cl.category==cl.true).sum()} acc={acc*100:.1f}% tokens={s['tokens']} tokens/ticket={s['tokens']/len(cl):.1f} transient={st['transient']} retry_lines={len(waits)} wait_range=({min(waits):.4f},{max(waits):.4f})")
            print("  statuses:",s["status_counts"],"redactions:",s["redactions"],"total",sum(s["redactions"].values()))
            if name=="zero-shot": print(pd.crosstab(cl.true,cl.category))
        redacted=[]
        from src.triage import redact
        for x in t.text:
            redacted.append(redact(x)[0])
        print("patterns found after redaction:",sum(len(p.findall(x)) for x in redacted for _,p,_ in PATTERNS))
        print("empty/long sent to model: none (guard before call); PII assertion on every prompt passed; est_tokens long ticket:",mock_llm.est_tokens(t.text[t.text.str.len()>1000].iloc[0]))
        for budget in (6000,):
            with contextlib.redirect_stdout(io.StringIO()):
                r,s=run_pipeline(prompt=ZERO,token_budget=budget)
            print(f"budget demo TOKEN_BUDGET={budget}: budget_exhausted={int((r.status=='budget_exhausted').sum())} tokens={s['tokens']}")
        with contextlib.redirect_stdout(io.StringIO()):
            r,s=run_pipeline(prompt=FEW,token_budget=100000)
        print("final rows",len(r),r.route.value_counts().to_dict(),r.priority.value_counts().to_dict())
    finally:
        mock_llm.chat=orig

def part_d_report():
    from src.policy_rag import PolicyRAG, evaluate
    print("\n--- Part D: retrieval ---")
    for ov in (10,0):
        rag=PolicyRAG(overlap=ov); d=evaluate(rag); a=d.iloc[:12]; u=d.iloc[12:]
        per={}
        for c in rag.chunks: per[c["source"]]=per.get(c["source"],0)+1
        print(f"overlap={ov}: chunks={len(rag.chunks)} {per}; doc-in-top3={int(a.retrieved_doc.sum())}/12 fact={int(a.fact_found.sum())}/12 fallback={int(u.fallback.sum())}/2")
        if ov==0: print("  failing answerable:",a[~(a.retrieved_doc&a.fact_found)].question.tolist())
    print("top3 for 'refund policy late delivery':",[h["source"] for h in PolicyRAG().retrieve("refund policy late delivery")])
    # agent
    from src import refund_agent as ra
    for p in (OUT/"agent_runs.jsonl",OUT/"refund_ledger.jsonl"):
        if p.exists(): p.unlink()
    if STATE.exists(): shutil.rmtree(STATE)
    STATE.mkdir()
    sc=pd.read_csv(ROOT/"data/raw/agent_scenarios.csv"); print("\n--- Part D: agent ---")
    with (OUT/"agent_runs.jsonl").open("a",encoding="utf-8") as f:
        for _,r in sc.iterrows():
            o=ra.run_agent(r.message,r.scenario_id)
            amt=next((s["observation"]["refund_inr"] for s in o["steps"] if s["tool"] in("issue_refund","escalate_to_human")),0.0)
            if o["status"] in("REFUND","ESCALATE"): assert abs(amt-r.expected_refund_inr)<0.01
            f.write(json.dumps({"scenario_id":r.scenario_id,"status":o["status"],"final_answer":o["final_answer"],"llm_calls":o["llm_calls"],"tokens_used":o["tokens_used"],"steps":o["steps"]})+"\n")
            print(f"| {r.scenario_id} | {o['status']} | {amt:.2f} | {o['llm_calls']} | {o['tokens_used']} |")
    s1=sc.iloc[0].message
    print("S1 token_limit=200 ->",ra.run_agent(s1,"S1",token_limit=200)["status"])
    print("S1 time_limit_seconds=0 ->",ra.run_agent(s1,"S1",time_limit_seconds=0)["status"])
    try: ra.run_agent(s1,"S1",run_id="s1_crash_demo",crash_after=2)
    except RuntimeError as e: print("simulated crash:",e)
    res=ra.resume_agent("s1_crash_demo")
    print("resume ->",res["status"],res["final_answer"],"tools:",[s["tool"] for s in res["steps"]])
    print("ledger lines:",sum(1 for _ in (OUT/"refund_ledger.jsonl").open()),"checkpoints:",len(list(STATE.glob("*.json"))))
