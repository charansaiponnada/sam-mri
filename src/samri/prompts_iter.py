"""Simulated corrective clicks for iterative prompt refinement (torch-free).

Given the previous prediction and the GT (oracle-error simulation, as in the box-prompt protocol),
place clicks on the *error regions*: positive clicks on false negatives (missed tumour), negative
clicks on false positives (over-segmentation). Within a connected error component the click is the
deepest interior pixel (max distance transform), which is the most informative single point.

Clicks alternate between the two error types, starting with whichever has more voxels, so a fixed
budget of 1/3/5 clicks covers both under- and over-segmentation instead of only one side.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def _component_points(mask):
    """One (x, y, size) point per connected component, largest component first."""
    lab, n = label(mask)
    if n == 0:
        return []
    dt = distance_transform_edt(mask)
    pts = []
    for i in range(1, n + 1):
        comp = lab == i
        y, x = np.unravel_index(int(np.argmax(np.where(comp, dt, -1.0))), dt.shape)
        pts.append((int(x), int(y), int(comp.sum())))
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts


def corrective_clicks(pred, gt, n, rng=None, spacing=None):
    """Return up to ``n`` corrective clicks as ``[(x, y, label), ...]`` (label 1=positive, 0=negative).

    Deterministic: the only randomness in the protocol is the prompt seed used by the caller; this
    function itself is pure. ``rng``/``spacing`` are accepted for API symmetry with the prompt
    simulators and to allow anisotropic distance transforms later.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    fn, fp = gt & ~pred, pred & ~gt
    if not fn.any() and not fp.any():
        return []
    queues = [
        (int(fn.sum()), _component_points(fn), 1),
        (int(fp.sum()), _component_points(fp), 0),
    ]
    queues.sort(key=lambda t: t[0], reverse=True)
    out = []
    idx = [0, 0]
    turn = 0
    while len(out) < n:
        took = False
        for k in range(2):
            side = (turn + k) % 2
            total, pts, lab = queues[side]
            if idx[side] < len(pts):
                x, y, _ = pts[idx[side]]
                idx[side] += 1
                out.append((x, y, lab))
                took = True
                turn = (side + 1) % 2
                break
        if not took:
            break
    return out


def accumulate(clicks, new):
    """Append ``new`` clicks to ``clicks`` while keeping the click history."""
    return list(clicks) + list(new)
