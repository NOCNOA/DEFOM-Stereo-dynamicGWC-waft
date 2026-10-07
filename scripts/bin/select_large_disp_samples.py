"""Select a GT-enriched TEST subset on CPU; do not interpret as random TEST performance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.stereo_datasets import SceneFlowDatasets
from core.utils.frame_utils import read_gen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/home/share/yijiayi/sceneflow/')
    p.add_argument('--output', required=True)
    p.add_argument('--max_samples', type=int, default=30)
    p.add_argument('--min_pixels', type=int, default=500)
    p.add_argument('--seed', type=int, default=1000)
    args = p.parse_args()
    if args.max_samples < 1 or args.min_pixels < 1:
        p.error('max_samples and min_pixels must be positive')
    dataset = SceneFlowDatasets(root=args.root, dstype='frames_finalpass', things_test=True)
    paths, counts = [], []
    scanned = 0
    for i in np.random.default_rng(args.seed).permutation(len(dataset)):
        scanned += 1
        gt = read_gen(dataset.disparity_list[int(i)][0])
        x = np.arange(gt.shape[1])[None, :]
        mask = np.isfinite(gt) & (gt >= 384) & (gt < 752) & (x - gt >= 0)
        count = int(np.count_nonzero(mask))
        if count >= args.min_pixels:
            paths.append(dataset.image_list[int(i)][0])
            counts.append(count)
            if len(paths) >= args.max_samples:
                break
    result = dict(selection='GT-enriched: >=min_pixels in 384-752 px with x-GT>=0',
                  seed=args.seed, min_pixels=args.min_pixels, scanned=scanned,
                  sample_paths=paths, large_pixel_counts=counts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f'Selected {len(paths)} images, {sum(counts)} large pixels; scanned {scanned}/{len(dataset)}. Saved {output}', flush=True)


if __name__ == '__main__':
    main()
