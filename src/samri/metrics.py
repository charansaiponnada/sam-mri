import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

EMPTY_HD95 = 373.13  # BraTS convention for "exactly one side empty"


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0  # both empty
    return float(2.0 * (pred & gt).sum() / s)


def _surface(m):
    st = generate_binary_structure(m.ndim, 1)
    return m & ~binary_erosion(m, structure=st, border_value=0)


def hd95(pred, gt, spacing=None, empty_penalty=EMPTY_HD95):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(empty_penalty)
    spacing = spacing or (1.0,) * pred.ndim
    sp, sg = _surface(pred), _surface(gt)
    d_to_gt = distance_transform_edt(~sg, sampling=spacing)
    d_to_pred = distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([d_to_gt[sp], d_to_pred[sg]])
    return float(np.percentile(d, 95))
