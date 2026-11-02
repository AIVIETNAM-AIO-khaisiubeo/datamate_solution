"""Per-handler precision on the DEV split only (sealed shops are excluded by run_eval.py) -> splits/handler_precision.json.
Laplace-smoothed: (correct+1)/(correct+wrong+2), so a handler needs many clean dev answers before the router trusts it
over a disagreeing agent."""
import pickle, json, os, collections
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = pickle.load(open(os.path.join(here, "res_train.pkl"), "rb"))
sealed = set(json.load(open(os.path.join(here, "splits", "split.json")))["sealed"])
assert not any(r["b"] in sealed for r in res), "sealed shops leaked into dev results"
c = collections.defaultdict(collections.Counter)
for r in res:
    c[r.get("handler")][r["score"]] += 1
prec = {h: round((v[1] + 1) / (v[1] + v[-1] + 2), 4) for h, v in c.items() if h}
json.dump(prec, open(os.path.join(here, "splits", "handler_precision.json"), "w"), indent=1, sort_keys=True)
print(json.dumps(prec, indent=1, sort_keys=True))
