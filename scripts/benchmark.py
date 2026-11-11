"""Submit a real corpus; include download, segmentation, queue and inference time."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('urls', help='Text file: one public image URL per line')
    p.add_argument('--base', default='http://localhost:8000')
    p.add_argument('--remove-background', action='store_true')
    args = p.parse_args()
    key = os.environ['API_KEY']
    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base + path, data=data,
            headers={'X-API-Key': key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)
    with open(args.urls, encoding='utf-8') as source:
        urls = [line.strip() for line in source if line.strip()]
    started = time.monotonic()
    ids = []
    for offset in range(0, len(urls), 1000):
        ids.extend(request('/v1/jobs', {'images': [dict(url=url, remove_background=args.remove_background)
                      for url in urls[offset:offset+1000]]})['job_ids'])
    pending, done, failed = set(ids), 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        while pending:
            jobs = list(pool.map(lambda jid: request('/v1/jobs/' + jid), pending))
            for job in jobs:
                if job['status'] in ('done', 'failed'):
                    pending.remove(job['id'])
                    done += job['status'] == 'done'
                    failed += job['status'] == 'failed'
            elapsed = time.monotonic() - started
            print(json.dumps(dict(done=done, failed=failed, pending=len(pending),
                                  seconds=round(elapsed, 2), images_per_second=round(done/elapsed, 2))))
            if pending:
                time.sleep(2)

if __name__ == '__main__':
    main()
