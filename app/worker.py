import asyncio
import gc
import io
import ipaddress
import json
import logging
import os
import socket
import time
import uuid
from urllib.parse import urlsplit, urljoin
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np
import torch
from PIL import Image, ImageOps
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from rembg import new_session, remove
from sentence_transformers import SentenceTransformer
from .common import REDIS_URL, STREAM, GROUP, TTL, setting

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('worker')
Image.MAX_IMAGE_PIXELS = setting('MAX_IMAGE_PIXELS', 25000000)

class PublicResolver(aiohttp.abc.AbstractResolver):
    """Validate and pin actual connection addresses, including every redirect."""
    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = {row[4][0] for row in infos}
        if not addresses or any(not ipaddress.ip_address(addr).is_global for addr in addresses):
            raise ValueError('Only public image hosts are allowed')
        return [dict(hostname=host, host=row[4][0], port=port, family=row[0], proto=0, flags=0)
                for row in infos]

    async def close(self):
        pass

def decode(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError('Image pixel limit exceeded')
        image = ImageOps.exif_transpose(source).convert('RGBA')
    image.thumbnail((setting('MAX_IMAGE_EDGE', 2048),) * 2)
    return image

def rgb(image):
    out = Image.new('RGB', image.size, 'white')
    out.paste(image, mask=image.getchannel('A'))
    return out

def validate_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Invalid image URL')
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return  # DNS names are validated and pinned by PublicResolver.
    if not address.is_global:
        raise ValueError('Only public image hosts are allowed')

FINISH = '''
if redis.call('XACK', KEYS[1], ARGV[1], ARGV[2]) == 1 then
 redis.call('SET', 'job:'..ARGV[3], ARGV[4], 'EX', ARGV[5])
 redis.call('DECR', 'pending')
 redis.call('XDEL', KEYS[1], ARGV[2])
end
return 1
'''

class Worker:
    def __init__(self):
        self.r = Redis.from_url(REDIS_URL, decode_responses=True)
        self.ready = asyncio.Queue(setting('READY_QUEUE_SIZE', 64))
        self.slots = asyncio.Semaphore(setting('PREPARE_CONCURRENCY', 16))
        self.gpu = asyncio.Lock()
        self.segmentation = asyncio.Lock()
        self.token = uuid.uuid4().hex
        self.tasks = set()
        self.batch_size = setting('BATCH_SIZE', 32)
        self.completed = 0
        self.started = time.monotonic()

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; refusing CPU fallback')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        self.model = SentenceTransformer(
            os.getenv('MODEL_ID', 'google/embeddinggemma-2'),
            revision=os.getenv('MODEL_REVISION', 'main'), device='cuda',
            config_kwargs={'audio_config': None},
            model_kwargs={'torch_dtype': torch.float32},
        )
        self.model.float().eval()
        if any(p.dtype != torch.float32 for p in self.model.parameters() if p.is_floating_point()):
            raise RuntimeError('Model is not FP32')
        device = os.getenv('REMBG_DEVICE', 'cuda')
        provider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        self.session = new_session(os.getenv('REMBG_MODEL', 'isnet-general-use'), providers=[provider])
        if provider not in self.session.inner_session.get_providers():
            raise RuntimeError('Requested rembg provider is unavailable')
        log.info('Loaded FP32 model without audio; GPU=%s', torch.cuda.get_device_name())

    async def finish(self, entry, job, **result):
        job.update(result)
        await self.r.eval(FINISH, 1, STREAM, GROUP, entry, job['id'], json.dumps(job), TTL)

    async def prepare(self, entry, jid):
        image = None
        job = {'id': jid}
        try:
            raw = await self.r.get('job:' + jid)
            if raw is None:
                raise ValueError('Missing job payload')
            job = json.loads(raw)
            job['status'] = 'processing'
            await self.r.set('job:' + jid, json.dumps(job))
            url = job['url']
            async with asyncio.timeout(setting('DOWNLOAD_TIMEOUT', 25)):
                for attempt in range(5):
                    validate_url(url)
                    async with self.http.get(url, allow_redirects=False) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            if attempt == 4 or 'Location' not in response.headers:
                                raise ValueError('Invalid or excessive redirects')
                            url = urljoin(url, response.headers['Location'])
                            continue
                        response.raise_for_status()
                        limit = setting('MAX_IMAGE_BYTES', 15728640)
                        if response.content_length and response.content_length > limit:
                            raise ValueError('Download size limit exceeded')
                        data = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            data.extend(chunk)
                            if len(data) > limit:
                                raise ValueError('Download size limit exceeded')
                        break
            image = await asyncio.to_thread(decode, data)
            del data
            if job['remove_background']:
                # Serialize CUDA access across ONNX Runtime and PyTorch.
                lock = self.gpu if os.getenv('REMBG_DEVICE', 'cuda') == 'cuda' else self.segmentation
                async with lock:
                    output = await asyncio.to_thread(remove, image, session=self.session)
                image.close()
                image = output.convert('RGBA')
                output.close()
                await asyncio.to_thread(image.save, Path('/data') / (jid + '.png'))
                job['image_url'] = '/v1/images/' + jid
            prepared = await asyncio.to_thread(rgb, image)
            await self.ready.put((entry, job, prepared))
        except Exception as exc:
            log.exception('Preparation failed for %s', jid)
            await self.finish(entry, job, status='failed', error=str(exc)[:400])
        finally:
            if image is not None:
                image.close()
            self.slots.release()

    async def dispatch(self):
        # Stable consumer name recovers all unfinished entries after a restart.
        recovering = True
        while True:
            await self.slots.acquire()
            rows = await self.r.xreadgroup(GROUP, 'single-gpu', {STREAM: '0' if recovering else '>'}, count=1, block=None if recovering else 1000)
            if not rows or not rows[0][1]:
                self.slots.release()
                recovering = False
                continue
            entry, fields = rows[0][1][0]
            if recovering:
                # Advance cursor to avoid scheduling the same pending entry twice.
                await self.recover_pending(entry, fields)
                recovering = False
                continue
            task = asyncio.create_task(self.prepare(entry, fields['id']))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def recover_pending(self, first, fields):
        cursor = first
        task = asyncio.create_task(self.prepare(first, fields['id']))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        while True:
            await self.slots.acquire()
            rows = await self.r.xreadgroup(GROUP, 'single-gpu', {STREAM: cursor}, count=1)
            if not rows or not rows[0][1]:
                self.slots.release()
                return
            cursor, fields = rows[0][1][0]
            task = asyncio.create_task(self.prepare(cursor, fields['id']))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    def encode(self, images):
        with torch.inference_mode():
            values = self.model.encode(
                [{'image': image} for image in images], batch_size=len(images),
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
                processing_kwargs={'image': {'max_soft_tokens': setting('VISION_TOKENS', 280)}},
            )
        if not np.isfinite(values).all():
            raise ValueError('Non-finite embeddings')
        return values.astype(np.float32).tolist()

    async def encode_batch(self, batch):
        try:
            async with self.gpu:
                vectors = await asyncio.to_thread(self.encode, [item[2] for item in batch])
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if len(batch) == 1:
                entry, job, image = batch[0]
                await self.finish(entry, job, status='failed', error='GPU out of memory for one image')
                image.close()
                return
            self.batch_size = max(1, len(batch) // 2)
            log.warning('OOM: reducing batch size to %d', self.batch_size)
            await self.encode_batch(batch[:self.batch_size])
            await self.encode_batch(batch[self.batch_size:])
            return
        except Exception as exc:
            if len(batch) > 1:
                middle = len(batch) // 2
                await self.encode_batch(batch[:middle])
                await self.encode_batch(batch[middle:])
                return
            entry, job, image = batch[0]
            await self.finish(entry, job, status='failed', error=str(exc)[:400])
            image.close()
            return
        for (entry, job, image), vector in zip(batch, vectors, strict=True):
            await self.finish(entry, job, status='done', embedding=vector,
                              dimensions=len(vector), model=os.getenv('MODEL_ID', 'google/embeddinggemma-2'),
                              precision='float32', vision_tokens=setting('VISION_TOKENS', 280))
            image.close()
            self.completed += 1

    async def embed_loop(self):
        while True:
            batch = [await self.ready.get()]
            deadline = asyncio.get_running_loop().time() + setting('BATCH_WAIT_MS', 20) / 1000
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.ready.get(), remaining))
                except asyncio.TimeoutError:
                    break
            await self.encode_batch(batch)

    async def heartbeat(self):
        while True:
            ok = await self.r.eval("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], 60) else return 0 end", 1, 'worker:lock', self.token)
            if not ok:
                raise RuntimeError('Worker lease lost; stopping to prevent duplicate GPU workers')
            await self.r.set('worker:alive', '1', ex=30)
            await self.r.hset('worker:stats', mapping={
                'completed_since_start': self.completed, 'ready_queue': self.ready.qsize(),
                'batch_size': self.batch_size, 'precision': 'float32',
                'images_per_second': self.completed / max(1, time.monotonic() - self.started),
                'gpu_allocated_bytes': torch.cuda.memory_allocated(),
            })
            await asyncio.sleep(5)

    async def cleanup(self):
        while True:
            def sweep():
                cutoff = time.time() - TTL
                for path in Path('/data').glob('*.png'):
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
            await asyncio.to_thread(sweep)
            await asyncio.sleep(60)

    async def run(self):
        Path('/data').mkdir(exist_ok=True)
        await asyncio.to_thread(self.load)
        if not await self.r.set('worker:lock', self.token, nx=True, ex=60):
            raise RuntimeError('Only one GPU worker is supported')
        try:
            await self.r.xgroup_create(STREAM, GROUP, id='0', mkstream=True)
        except ResponseError as exc:
            if 'BUSYGROUP' not in str(exc):
                raise
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), limit=setting('PREPARE_CONCURRENCY', 16), use_dns_cache=False)
        async with aiohttp.ClientSession(connector=connector, trust_env=False,
                timeout=aiohttp.ClientTimeout(total=setting('DOWNLOAD_TIMEOUT', 25))) as self.http:
            async with asyncio.TaskGroup() as group:
                for task in (self.dispatch, self.embed_loop, self.heartbeat, self.cleanup):
                    group.create_task(task())

if __name__ == '__main__':
    async def main():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=setting('PREPARE_CONCURRENCY', 16) + 2))
        await Worker().run()
    asyncio.run(main())
