"""Read video timestamps and attach lossless frames referenced by human points."""
import base64
from functools import lru_cache
import json
import math
import subprocess


@lru_cache(maxsize=64)
def frame_times(video, mtime_ns, size):
    result = subprocess.run(['ffprobe', '-threads', '0', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True, timeout=120)
    times = [float(row['best_effort_timestamp_time']) for row in json.loads(result.stdout)['frames']]
    if not times or any(not math.isfinite(t) for t in times):
        raise ValueError('Video lacks usable frame timestamps')
    times = [time-times[0] for time in times]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Video timestamps are not strictly increasing')
    return times


def frame_reference(frame):
    """Expose the time and cue association; keep the decoder index internal."""
    return {'time': frame['time'], 'cues': frame['cues']}


def evidence_images(prepared):
    parts = []
    for frame in prepared.get('evidence_frames', []):
        index = frame['frame_index']
        result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(prepared['video']),
            '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'png', '-'],
            capture_output=True, check=True, timeout=120)
        if not result.stdout.startswith(b'\x89PNG'):
            raise ValueError('Failed to extract evidence frame')
        parts.extend([{'type':'text','text':'Evidence frame: '+json.dumps(frame_reference(frame))},
                      {'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                        base64.b64encode(result.stdout).decode()}}])
    return parts
