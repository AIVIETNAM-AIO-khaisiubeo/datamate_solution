"""Count parameters of local checkpoints from safetensors headers (no torch needed) and check the 15B budget.
Usage: python tools/count_params.py /opt/models/gen /opt/models/ver ..."""
import sys, os, json, struct, glob

def count(path):
    n = 0
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            p = 1
            for d in v["shape"]:
                p *= d
            # packed int4 (AWQ/GPTQ) stores 8 weights per int32: count logical params
            if v["dtype"] in ("I32",) and (k.endswith("qweight") or k.endswith("qzeros")):
                p *= 8 if k.endswith("qweight") else 0
            if k.endswith(("scales", "g_idx")):
                p = 0
            n += p
    return n

tot = 0
for p in sys.argv[1:]:
    c = count(p); tot += c
    print(f"{p}: {c/1e9:.2f}B")
print(f"TOTAL {tot/1e9:.2f}B  ->  {'OK' if tot <= 15e9 else 'OVER 15B LIMIT'}")
