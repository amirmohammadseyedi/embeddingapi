import asyncio
import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, Field
from redis.asyncio import Redis
from .common import REDIS_URL, STREAM, TTL, setting

r = Redis.from_url(REDIS_URL, decode_responses=True)

@asynccontextmanager
async def lifespan(app):
    if not os.getenv('API_KEY') or os.getenv('API_KEY') == 'change-this-to-a-long-random-secret':
        raise RuntimeError('Set a strong API_KEY in .env')
    await r.ping()
    yield
    await r.aclose()

app = FastAPI(
    title='Image Embedding Service',
    version='1.0.0',
    description=(
        'Embed images with EmbeddingGemma 2 in FP32, optionally removing the background. '
        'Use Authorize to enter your API_KEY (X-API-Key header). '
        'For high volume, submit up to 1000 images to /v1/jobs and poll each returned ID. '
        'GPU batch size is configured separately; the default is 32.'
    ),
    docs_url='/docs', redoc_url='/redoc', openapi_url='/openapi.json',
    swagger_ui_parameters={'displayRequestDuration': True, 'defaultModelsExpandDepth': 1},
    openapi_tags=[
        {'name': 'Embedding', 'description': 'Single-image processing and bulk submissions.'},
        {'name': 'Results', 'description': 'Job results and background-removed PNG downloads.'},
        {'name': 'Monitoring', 'description': 'Readiness and queue/worker statistics.'},
    ],
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name='X-API-Key', scheme_name='ApiKeyAuth', auto_error=False,
                             description='Enter the API_KEY configured in your .env file.')

async def auth(x_api_key: str | None = Security(api_key_header)):
    if not x_api_key or not secrets.compare_digest(x_api_key, os.getenv('API_KEY', '')):
        raise HTTPException(401, 'Invalid API key')

class ImageRequest(BaseModel):
    url: HttpUrl = Field(description='Public HTTP(S) image URL.', examples=['https://example.com/product.jpg'])
    remove_background: bool = Field(default=False, description='Remove background before embedding; also save a transparent PNG.')

class BatchRequest(BaseModel):
    images: list[ImageRequest] = Field(
        min_length=1, max_length=1000,
        description='Up to 1000 images; background removal is selected independently for each image.',
        examples=[[
            {'url': 'https://example.com/a.jpg', 'remove_background': True},
            {'url': 'https://example.com/b.jpg', 'remove_background': False},
        ]],
    )

# Atomically reserve capacity and publish the entire request.
SUBMIT = '''
local n = tonumber(ARGV[1])
local current = tonumber(redis.call('GET', 'pending') or '0')
if current + n > tonumber(ARGV[2]) then return 0 end
for i = 1, n do
 local base = 3 + (i-1)*2
 redis.call('SET', 'job:'..ARGV[base], ARGV[base+1])
 redis.call('XADD', KEYS[1], '*', 'id', ARGV[base])
end
redis.call('INCRBY', 'pending', n)
return 1
'''

async def submit(images):
    ids, args = [], [len(images), setting('MAX_PENDING', 10000)]
    for image in images:
        jid = uuid.uuid4().hex
        ids.append(jid)
        args.extend([jid, json.dumps(dict(id=jid, status='queued', **image.model_dump(mode='json')))])
    if not await r.eval(SUBMIT, 1, STREAM, *args):
        raise HTTPException(429, 'Queue full; retry later', headers={'Retry-After': '5'})
    return ids

@app.post('/v1/jobs', status_code=202, dependencies=[Depends(auth)], tags=['Embedding'],
          summary='Queue up to 1000 images', responses={429: {'description': 'Queue full; retry later.'}})
async def batch(body: BatchRequest):
    """Returns job_ids immediately. Poll GET /v1/jobs/{jid} for each image's result."""
    return {'job_ids': await submit(body.images)}

@app.post('/v1/embed', dependencies=[Depends(auth)], tags=['Embedding'], summary='Embed one image')
async def embed(body: ImageRequest):
    """Wait up to 60 seconds for a result, then return a job ID and poll_url. Processing continues in the queue."""
    jid = (await submit([body]))[0]
    for _ in range(120):
        job = json.loads(await r.get('job:' + jid))
        if job['status'] in ('done', 'failed'):
            return job
        await asyncio.sleep(0.5)
    return {'id': jid, 'status': 'processing', 'poll_url': '/v1/jobs/' + jid}

@app.get('/v1/jobs/{jid}', dependencies=[Depends(auth)], tags=['Results'], summary='Get job status and embedding',
         responses={404: {'description': 'Job not found or result expired.'}})
async def result(jid: str):
    """Statuses: queued, processing, done, failed. A done job includes a normalized float32 embedding."""
    value = await r.get('job:' + jid)
    if value is None:
        raise HTTPException(404, 'Job not found or expired')
    return json.loads(value)

@app.get('/v1/images/{jid}', dependencies=[Depends(auth)], tags=['Results'], summary='Download transparent PNG',
         response_class=FileResponse,
         responses={200: {'description': 'PNG available only for background-removal jobs.',
                          'content': {'image/png': {'schema': {'type': 'string', 'format': 'binary'}}}},
                    404: {'description': 'No output PNG or file expired.'}})
async def image(jid: str):
    if len(jid) != 32 or any(c not in '0123456789abcdef' for c in jid):
        raise HTTPException(404)
    path = Path('/data') / (jid + '.png')
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='image/png')

@app.get('/health', tags=['Monitoring'], summary='Check API and GPU worker readiness',
         responses={503: {'description': 'GPU worker starting or unavailable.'}})
async def health():
    await r.ping()
    alive = bool(await r.get('worker:alive'))
    if not alive:
        raise HTTPException(503, 'GPU worker starting or unavailable')
    return {'ready': True}

@app.get('/v1/stats', dependencies=[Depends(auth)], tags=['Monitoring'], summary='Read queue and GPU worker statistics')
async def stats():
    return {'pending': int(await r.get('pending') or 0), 'worker': await r.hgetall('worker:stats')}
