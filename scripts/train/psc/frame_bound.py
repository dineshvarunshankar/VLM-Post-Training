"""Measure the exact training sampler's frame bound from the selected inventory."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers.video_utils import VideoMetadata
from scripts.train.shared.data import DATASETS,atomic_json
from scripts.train.shared.run_config import load_run,compile_run
from scripts.train.shared.video import sample_capped_frames,source_video_config
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--output', required=True)
args=parser.parse_args()
plan=compile_run(load_run(args.config))
c=plan['run'];video=plan['experiment']['video'];media=Path(c['paths']['media_root']);splits=Path(c['paths']['split_dir'])
clips=set();hashes={}
for split in ('train','validation','test'):
 p=splits/(split+'.jsonl');raw=p.read_bytes();hashes[split]=hashlib.sha256(raw).hexdigest()
 for line in raw.decode().splitlines():
  row=json.loads(line)
  if row['variant'] in c['dataset']['sources']:clips.add((row['variant'],row['clip']))
def measure(item):
 variant,clip=item;p=media/DATASETS[variant]/clip/'all_annotations/meta.json';raw=p.read_bytes();meta=json.loads(raw)
 count=len(sample_capped_frames(VideoMetadata(total_num_frames=meta['n_frames'],fps=meta['fps']),fps_cap=source_video_config(video,variant)['fps']))
 return {'variant':variant,'clip':clip,'sampled_frames':count,'n_frames':meta['n_frames'],'fps':meta['fps'],'meta_sha256':hashlib.sha256(raw).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(measure,sorted(clips)))
report={'profile_video_frames':max(2,max(r['sampled_frames'] for r in rows)),'clips':len(rows),'split_sha256':hashes,'maximum_cases':sorted(rows,key=lambda r:r['sampled_frames'],reverse=True)[:8],'source_maxima':{s:max(r['sampled_frames'] for r in rows if r['variant']==s) for s in c['dataset']['sources']},'method':'actual sample_capped_frames from annotation FPS/frame-count; no video decoding or truncation'}
atomic_json(Path(args.output),report)
print(json.dumps(report),flush=True)
