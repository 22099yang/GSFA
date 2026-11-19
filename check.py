import json, numpy as np
p = json.load(open(r"D:\temp\3DGS\gaussian_splatting\Data\Bottle\sparse\0\depth_params.json"))
items = sorted(p.items(), key=lambda kv: kv[1]["scale"])
med = np.median([v["scale"] for _, v in items])
print("median", med)
print("分位数 10/25/50/75/90:", np.percentile([v["scale"] for _, v in items], [10,25,50,75,90]))
for k, v in items:
    r = v["scale"] / med
    flag = "  <-- 异常" if r > 1.5 or r < 0.67 else ""
    print(f"{k:30s} scale={v['scale']:8.2f}  ratio={r:5.2f}  offset={v['offset']:+.4f}{flag}")