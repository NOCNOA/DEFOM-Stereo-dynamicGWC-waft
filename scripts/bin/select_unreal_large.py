"""Build a model-independent, scene-balanced Unreal large-disparity manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/share/zhaobofan/UnrealStereo4K'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-visible-large', type=int, default=5000,
                   help='minimum stride-4 pixels with 384<=GT<752 and LR error<=1 px')
    p.add_argument('--per-scene', type=int, default=20)
    p.add_argument('--minimum-frame-gap', type=int, default=20)
    p.add_argument('--frame-stride', type=int, default=1,
                   help='scan only frame numbers divisible by this value')
    p.add_argument('--target-total', type=int, default=0,
                   help='cap the balanced manifest after selection; zero keeps all')
    args = p.parse_args()
    candidates = []
    unreadable = 0
    scanned = 0
    for scene in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for left in sorted((scene/'Image0').glob('*.png')):
            if int(left.stem) % args.frame_stride:
                continue
            scanned += 1
            if scanned % 250 == 0:
                print(f'scanned={scanned} candidates={len(candidates)} unreadable={unreadable}', flush=True)
            right = scene/'Image1'/left.name
            disp0 = scene/'Disp0'/f'{left.stem}.npy'
            disp1 = scene/'Disp1'/f'{left.stem}.npy'
            if not right.is_file() or not disp0.is_file() or not disp1.is_file():
                unreadable += 1
                continue
            try:
                dl = np.load(disp0, mmap_mode='r', allow_pickle=False)
                if dl.shape != (2160,3840):
                    raise ValueError('shape')
                d = np.asarray(dl[::4,::4])
                x = np.arange(0,3840,4,dtype=np.float32)[None]
                xr = x-d
                base = np.isfinite(d)&(d>=384)&(d<752)&(xr>=0)&(xr<=3839)
                # Most frames have no useful large disparity. Avoid touching the
                # 4K right GT unless the cheaper left-only screen can pass.
                if int(base.sum()) < args.minimum_visible_large:
                    continue
                dr = np.load(disp1, mmap_mode='r', allow_pickle=False)
                if dr.shape != dl.shape:
                    raise ValueError('shape')
                rr = dr[::4]
                x0 = np.floor(np.where(base,xr,0)).astype(np.int32)
                x1 = np.minimum(x0+1,3839)
                alpha = xr-x0
                y = np.arange(d.shape[0])[:,None]
                sampled = (1-alpha)*rr[y,x0]+alpha*rr[y,x1]
                visible = base & np.isfinite(sampled)&(sampled>0)&(np.abs(sampled-d)<=1)
                count = int(visible.sum())
                if count < args.minimum_visible_large:
                    continue
                with Image.open(left) as image:
                    image.thumbnail((320,180))
                    array=np.asarray(image.convert('RGB'))
                near_black=float((array.max(-1)<=3).mean())
                if near_black>=.95:
                    continue
            except (OSError,ValueError,EOFError):
                unreadable += 1
                continue
            candidates.append(dict(scene=scene.name,frame=int(left.stem),left=str(left),
                                   visible_large_stride4=count,near_black_fraction=near_black))
    selected=[]
    for scene in sorted({row['scene'] for row in candidates}):
        rows=sorted((row for row in candidates if row['scene']==scene),
                    key=lambda row:(-row['visible_large_stride4'],row['frame']))
        chosen=[]
        for row in rows:
            if all(abs(row['frame']-other['frame'])>=args.minimum_frame_gap for other in chosen):
                chosen.append(row)
            if len(chosen)==args.per_scene:
                break
        selected.extend(sorted(chosen,key=lambda row:row['frame']))
    if args.target_total and len(selected) > args.target_total:
        # Round-robin by scene so the cap cannot be filled by one scene.
        groups={scene:[row for row in selected if row['scene']==scene]
                for scene in sorted({row['scene'] for row in selected})}
        selected=[]
        while len(selected)<args.target_total and any(groups.values()):
            for scene in groups:
                if groups[scene] and len(selected)<args.target_total:
                    selected.append(groups[scene].pop(0))
    payload=dict(selection='GT-only: stride-4 384<=disp<752, x-d in bounds, right-GT LR error<=1 px; image readable and not near-black',
                 root=str(args.root),minimum_visible_large_stride4=args.minimum_visible_large,
                 per_scene_cap=args.per_scene,minimum_frame_gap=args.minimum_frame_gap,
                 frame_stride=args.frame_stride,target_total=args.target_total,
                 unreadable_or_invalid=unreadable,candidate_count=len(candidates),
                 selected_count=len(selected),scene_counts={s:sum(r['scene']==s for r in selected) for s in sorted({r['scene'] for r in selected})},
                 sample_paths=[row['left'] for row in selected],samples=selected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({key:payload[key] for key in ('unreadable_or_invalid','candidate_count','selected_count','scene_counts')},indent=2))


if __name__=='__main__':
    main()
