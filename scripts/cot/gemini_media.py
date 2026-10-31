"""Upload prepared MP4 files and reuse active Gemini Files API references."""
import asyncio
from contextlib import asynccontextmanager
import fcntl
import hashlib
import json
import os


REQUEST_TIMEOUT_MS = 60_000
FILE_OPERATION_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 2


@asynccontextmanager
async def upload_lock(path):
    with path.open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def upload_video(source, source_hash, cache_dir):
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source_hash:
            raise ValueError('Video changed before upload')
    return asyncio.run(_upload_video(source, source_hash, cache_dir))


async def _upload_video(source, source_hash, cache_dir):
    from google import genai
    from google.genai import errors

    # Namespaced by credential digest, never the credential itself.
    api_key = os.environ['GEMINI_API_KEY']
    account = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    folder = cache_dir / account
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / f'{source_hash}.json'
    deadline = asyncio.get_running_loop().time() + FILE_OPERATION_TIMEOUT_SECONDS
    async with asyncio.timeout_at(deadline), upload_lock(folder / f'{source_hash}.lock'):
        async with genai.Client(api_key=api_key, http_options={'timeout': REQUEST_TIMEOUT_MS}).aio as client:
            if record_path.exists():
                try:
                    cached = json.loads(record_path.read_text())
                    remote = await client.files.get(name=cached['name'])
                    if remote.state.name == 'ACTIVE' and remote.uri:
                        return remote.uri
                except errors.ClientError as error:
                    if error.code not in (404, 410):
                        raise
                except (ValueError, KeyError):
                    pass
            remote = await client.files.upload(file=source, config={'mime_type': 'video/mp4'})
            with source.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != source_hash:
                    raise ValueError('Video changed during upload')
            while remote.state.name == 'PROCESSING':
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                remote = await client.files.get(name=remote.name)
            if remote.state.name != 'ACTIVE' or not remote.uri:
                raise ValueError('Gemini did not activate the uploaded video')
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('Gemini file upload and activation deadline exceeded')
            # Protected by upload_lock; interrupted cache writes are safe to discard.
            record_path.write_text(json.dumps({'name': remote.name, 'uri': remote.uri,
                                              'source_sha256': source_hash}) + '\n')
            return remote.uri
