import sys, os, numpy as np, cv2
sys.path.insert(0, "utils")
from read_write_model import read_model, qvec2rotmat

base = r"D:\temp\3DGS\gaussian_splatting\Data\Bottle"
cams, imgs, pts = read_model(os.path.join(base, "sparse", "0"), ext=".txt")

P = np.zeros((max(pts.keys()) + 1, 3))
for k, v in pts.items():
    P[k] = v.xyz

for im in list(imgs.values())[:3]:
    m = im.point3D_ids >= 0
    n = int(m.sum())
    if n > 0:
        z = (P[im.point3D_ids[m]] @ qvec2rotmat(im.qvec).T + im.tvec)[:, 2]
        inv = 1.0 / z
        rng = (float(inv.min()), float(inv.max()))
    else:
        rng = None
    name = im.name.rsplit(".", 1)[0]
    d = cv2.imread(os.path.join(base, "dpimage", name + ".png"), cv2.IMREAD_UNCHANGED)
    info = None if d is None else (d.dtype, d.shape, int(d.min()), int(d.max()))
    print(im.name, "| 稀疏点数:", n, "| COLMAP逆深度范围:", rng, "| 深度图:", info)