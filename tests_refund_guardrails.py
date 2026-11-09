"""Guardrail tests: invalid refund attempts must be rejected and must not touch the ledger."""
import sqlite3, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src import refund_agent as ra
LEDGER = ra.OUT / "refund_ledger.jsonl"
con = sqlite3.connect(ra.DB)
def pick(q): return con.execute(q).fetchone()
canc = pick("select order_id,order_amount from orders where status='cancelled' limit 1")
ontime = pick("select order_id,order_amount from orders where status='completed' and delivery_minutes<=45 limit 1")
big = pick("select order_id,order_amount from orders where status='completed' and delivery_minutes>45 and order_amount>1000 limit 1")
small = pick("select order_id,order_amount from orders where status='completed' and delivery_minutes>45 and order_amount<=1000 limit 1")
before = LEDGER.read_text() if LEDGER.exists() else ""
tests = {
 "cancelled order cannot be refunded": ra.issue_refund(canc[0], canc[1]),
 "on-time delivery cannot be refunded": ra.issue_refund(ontime[0], ontime[1]),
 "refund above 1000 cannot be issued automatically": ra.issue_refund(big[0], big[1]),
 "refund of 1000 or less cannot be escalated": ra.escalate_to_human(small[0], small[1]),
 "wrong refund amount is rejected": ra.issue_refund(small[0], small[1] + 5),
}
for name, res in tests.items():
    assert "error" in res, f"FAILED: {name}"; print("PASS:", name, "->", res["error"])
after = LEDGER.read_text() if LEDGER.exists() else ""
assert before == after, "ledger changed by an invalid attempt"
ok = ra.issue_refund(small[0], small[1]); assert ok["refund_inr"] == small[1]
print("PASS: valid refund accepted; ledger untouched by the 5 invalid attempts")
