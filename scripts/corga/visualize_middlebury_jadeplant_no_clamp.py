"""Render a focused Jadeplant no-clamp prediction and error figure."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from core.utils import frame_utils

root = Path('diagnostics/large_delta_eval/jadeplant_visualization')
scene = Path('/home/share/yijiayi/Middlebury/MiddEval3/trainingF/Jadeplant')
left = np.asarray(Image.open(scene / 'im0.png').convert('L'))
gt = np.asarray(frame_utils.readDispMiddlebury(str(scene / 'disp0GT.pfm'))[0], dtype=np.float32)
pred = np.load(root / 'no_clamp/Jadeplant_pred.npy')
valid = np.isfinite(gt) & (gt > 0) & (gt < 768)
error = np.abs(pred - gt)
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle('Jadeplant | no clamp | EPE 54.05 px | Out2 54.32%', fontsize=17)
axes[0, 0].imshow(left); axes[0, 0].set_title('Left image')
for ax, data, title, cmap, maximum in [
    (axes[0, 1], gt, 'Ground truth disparity', 'gray', 600),
    (axes[1, 0], pred, 'No-clamp prediction', 'gray', 600),
    (axes[1, 1], error, 'Absolute error', 'gray', 250),
]:
    colors = plt.get_cmap(cmap).copy(); colors.set_bad('#252525')
    im = ax.imshow(np.ma.masked_where(~valid, data), cmap=colors, vmin=0, vmax=maximum)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.035, pad=.01, label='pixels')
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
output = root / 'jadeplant_no_clamp_grayscale.png'
fig.savefig(output, dpi=150)
plt.close(fig)
print(output)
