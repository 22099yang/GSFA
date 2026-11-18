import sys, os, numpy as np, cv2
sys.path.insert(0, "utils")
from read_write_model import read_model

base = r"D:\temp\3DGS\gaussian_splatting\Data\Bottle"
cams, imgs, pts = read_model(os.path.join(base, "sparse", "0"), ext=".txt")

for im in list(imgs.values())[:3]:
    cam = cams[im.camera_id]
    name = im.name.rsplit(".", 1)[0]
    d = cv2.imread(os.path.join(base, "dpimage", name + ".png"), cv2.IMREAD_UNCHANGED)
    d2 = d[..., 0] if d.ndim == 3 else d
    m = im.point3D_ids >= 0
    xy = im.xys[m]
    print(im.name, "| 相机宽高:", cam.width, cam.height, "| 深度图 H,W:", d2.shape)
    print("  稀疏点 x 范围:", xy[:, 0].min(), xy[:, 0].max(), " y 范围:", xy[:, 1].min(), xy[:, 1].max())
    sx = d2.shape[1] / cam.width
    sy = d2.shape[0] / cam.height
    x = np.clip((xy[:, 0] * sx).astype(int), 0, d2.shape[1] - 1)
    y = np.clip((xy[:, 1] * sy).astype(int), 0, d2.shape[0] - 1)
    v = d2[y, x].astype(float)
    med = np.median(v)
    print("  采样深度值: 最小", v.min(), "最大", v.max(), "中位数", med, "MAD", np.median(np.abs(v - med)))