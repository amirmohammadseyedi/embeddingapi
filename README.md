# Image embedding service

FastAPI + Redis Streams + a single CUDA worker. Each image optionally passes through rembg before EmbeddingGemma 2. Audio encoder is disabled. Embeddings use float32; TF32 and mixed precision are disabled. Output vectors are L2 normalized and have 768 dimensions.

## Start on the A100 Linux server

Install Docker Engine, Docker Compose with `gpus` support, NVIDIA driver and NVIDIA Container Toolkit. Verify GPU access with an NVIDIA CUDA container before deployment.

```sh
cp .env.example .env
# Edit .env: set API_KEY and, if required by the model repository, HF_TOKEN.
docker compose up -d --build
docker compose logs -f worker
```

The first run downloads model weights. `/health` returns 503 until the model is loaded. Open `http://localhost:8000/docs` for the API schema. All processing/result/image/stats endpoints require the `X-API-Key` header. Expose publicly only behind a TLS reverse proxy.

### Swagger UI

Visit `/docs`, click **Authorize**, and enter the `API_KEY` from `.env` without any prefix. Then choose an endpoint, click **Try it out**, edit the sample JSON (including real public image URLs and `remove_background`), and click **Execute**. The key is sent as `X-API-Key` to protected endpoints. The schema is available at `/openapi.json`; alternative read-only documentation is at `/redoc`. Swagger displays request durations, single/bulk examples and PNG download responses. Authentication is not persisted across page reloads.

## Single image

```sh
curl http://localhost:8000/v1/embed \
  -H 'X-API-Key: YOUR_SECRET' -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/product.jpg","remove_background":true}'
```

Returns the embedding and an authenticated PNG download path when background removal was enabled. With `false`, no background removal or output PNG is performed. After 60 seconds, this endpoint returns a job ID to poll; it does not cancel processing.

## High volume (recommended)

```sh
curl http://localhost:8000/v1/jobs \
  -H 'X-API-Key: YOUR_SECRET' -H 'Content-Type: application/json' \
  -d '{"images":[{"url":"https://example.com/a.jpg","remove_background":true},{"url":"https://example.com/b.jpg","remove_background":false}]}'
curl -H 'X-API-Key: YOUR_SECRET' http://localhost:8000/v1/jobs/JOB_ID
curl -H 'X-API-Key: YOUR_SECRET' http://localhost:8000/v1/stats
curl -H 'X-API-Key: YOUR_SECRET' http://localhost:8000/v1/images/JOB_ID -o result.png
```

Up to 1000 images per submission, 10000 pending by default. Queue overflow returns 429. Keep batches of submissions below the pending limit; poll finished work before submitting more. Input queue and pending payloads survive restarts using Redis AOF (default durability may lose approximately one second on host failure). Pending work is retried after worker restart. Completed results expire after RESULT_TTL (24 hours); PNG cleanup runs every minute. Download/store results before expiry. Redis storage errors return server errors rather than silently dropping work. Only one worker is supported; do not scale the worker service. No per-job automatic network retry; failed jobs can be resubmitted by clients.

## Tune throughput on your real corpus

Defaults target the specified 32-core / 128 GB / A100 40 GB machine, but have not been benchmarked on it. FP32 costs more compute/memory than BF16. GPU utilization is not equivalent to useful images/sec.

1. Warm up and then benchmark several thousand representative images separately with and without background removal.
2. Try BATCH_SIZE=16,32,64,128. Start at 32. CUDA OOM during embedding automatically splits the batch and reduces the batch target. Other batch errors are isolated by splitting, so one bad image does not fail the whole batch.
3. Try PREPARE_CONCURRENCY=8,16,24 and READY_QUEUE_SIZE=32,64,128. These are bounded decoded-image buffers; large values can consume substantial RAM. MAX_IMAGE_EDGE=2048 caps processing resolution including saved PNG dimensions. Change if full-resolution output is required; RAM and segmentation cost will rise.
4. Compare REMBG_DEVICE=cuda and cpu. GPU inference is serialized across rembg and embedding to avoid uncontrolled contention. CPU mode is supported but segmentation is currently serialized too; this is a starting point, not an optimized multi-session CPU pool.
5. VISION_TOKENS=280 is the default. Try 140 or 70 for speed only after checking retrieval quality. Use identical preprocessing and vision budget for corpus and query images. Transparent pixels are composited on white for embeddings, while rembg output PNG retains alpha.
6. Watch `nvidia-smi`, RAM, network and `/v1/stats`. Idle GPU with an empty ready queue means preparation/download/segmentation is limiting throughput. A continuously full ready queue means embedding is limiting it. Stats report successful images/sec since worker start, including warmup/idle time; use the benchmark for controlled runs.

Apply .env changes with `docker compose up -d --force-recreate`. Model ID and revision are recorded/configurable; pin MODEL_REVISION to a model commit for reproducibility. Python dependency ranges permit installing current model support, so freeze resolved versions after validating the first build (`docker compose exec worker pip freeze`). The image build runs `pip check`; runtime model support still requires an actual startup/smoke test on the server.

```sh
export API_KEY='YOUR_SECRET'
python scripts/benchmark.py urls.txt --base http://SERVER:8000
python scripts/benchmark.py urls.txt --base http://SERVER:8000 --remove-background
```

Benchmark performs real API calls and downloads completed embedding JSON; use a controlled public image host for repeatability. It is not a pure model-only benchmark.

## Validation and operational limits

Public HTTP(S) URLs only. Private/reserved DNS resolutions, direct private IPs, credentials in URLs, excessive redirects, byte/pixel limits and download deadlines are checked. Connections use a resolver that pins validated addresses. Queue capacity is reserved atomically. No automatic GPU fallback to CPU. A Redis lease prevents a second worker from consuming concurrently, though model loading happens before acquiring the lease. Run one Compose stack per GPU.

Source references: [Google image inputs](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers), [precision and audio disabling](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [rembg](https://github.com/danielgatis/rembg).
