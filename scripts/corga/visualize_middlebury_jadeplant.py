"""Render the saved Jadeplant predictions from the matched Middlebury F runs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from core.utils import frame_utils

root = Path('diagnostics/large_delta_eval/jadeplant_visualization')
scene = Path('/home/share/yijiayi/Middlebury/MiddEval3/trainingF/Jadeplant')
left = np.asarray(Image.open(scene / 'im0.png').convert('RGB'))
gt = np.asarray(frame_utils.readDispMiddlebury(str(scene / 'disp0GT.pfm'))[0], dtype=np.float32)
base = np.load(root / 'baseline/Jadeplant_pred.npy')
free = np.load(root / 'no_clamp/Jadeplant_pred.npy')
valid = np.isfinite(gt) & (gt > 0) & (gt < 768)
if left.shape[:2] != gt.shape or base.shape != gt.shape or free.shape != gt.shape:
    raise ValueError('Input shapes do not match')
err_base = np.abs(base - gt)
err_free = np.abs(free - gt)
fig, axes = plt.subplots(3, 2, figsize=(17, 19), constrained_layout=True)
fig.suptitle('Middlebury trainingF / Jadeplant   |   two-stage 200k, coarse 6, refine 32', fontsize=18)
axes[0, 0].imshow(left)
axes[0, 0].set_title('Left image')
disp_cmap = plt.get_cmap('turbo').copy(); disp_cmap.set_bad('#252525')
err_cmap = plt.get_cmap('magma').copy(); err_cmap.set_bad('#252525')
images = [
    (axes[0, 1], gt, 'Ground truth', disp_cmap, 0, 600),
    (axes[1, 0], base, 'Original limit  |  EPE 85.17 px', disp_cmap, 0, 600),
    (axes[1, 1], free, 'No clamp  |  EPE 54.05 px', disp_cmap, 0, 600),
    (axes[2, 0], err_base, 'Absolute error: original limit', err_cmap, 0, 250),
    (axes[2, 1], err_free, 'Absolute error: no clamp', err_cmap, 0, 250),
]
for ax, array, title, cmap, low, high in images:
    im = ax.imshow(np.ma.masked_where(~valid, array), cmap=cmap, vmin=low, vmax=high)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.035, pad=.01, label='pixels')
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
output = root / 'jadeplant_comparison.png'
fig.savefig(output, dpi=150)
plt.close(fig)
print(output)
