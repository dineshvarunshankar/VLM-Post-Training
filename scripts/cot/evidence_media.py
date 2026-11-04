"""Read video timestamps and attach lossless frames referenced by human points."""
import base64
from functools import lru_cache
import json
import math
import subprocess

from source_cache import cached_files


def video_duration(video):
    """Read duration from the prepared video when copied metadata omits it."""
    def probe():
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration', '-of', 'json', str(video)],
            capture_output=True, text=True, check=True, timeout=120)
        streams = json.loads(result.stdout).get('streams', [])
        try:
            duration = float(streams[0]['duration'])
        except (IndexError, KeyError, TypeError, ValueError):
            raise ValueError('Prepared video has no usable duration') from None
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Prepared video has no usable duration')
        return duration
    return cached_files([video], 'video_duration', probe)


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


def _frame_png(video, index):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(video),
        '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'png', '-'],
        capture_output=True, check=True, timeout=120)
    if not result.stdout.startswith(b'\x89PNG'):
        raise ValueError('Failed to extract evidence frame')
    return 'data:image/png;base64,' + base64.b64encode(result.stdout).decode()


def evidence_images(prepared):
    parts = []
    video = prepared['video']
    for frame in prepared.get('evidence_frames', []):
        index = frame['frame_index']
        uri = cached_files([video], ('frame_png', index), lambda: _frame_png(video, index))
        parts.extend([{'type': 'text', 'text': 'Evidence frame: ' + json.dumps(frame_reference(frame))},
                      {'type': 'image_url', 'image_url': {'url': uri}}])
    return parts
