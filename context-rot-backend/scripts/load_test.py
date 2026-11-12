"""Measure real SSE first-content timing. Offline results are NOT Gemini benchmarks."""
import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import httpx


async def account(client):
    name="bench_"+uuid4().hex[:12];password=uuid4().hex
    response=await client.post('/auth/signup',json=dict(username=name,email=name+'@example.com',password=password));response.raise_for_status()
    response=await client.post('/auth/login',json=dict(username=name,password=password));response.raise_for_status()
    return {'Authorization':'Bearer '+response.json()['access_token']}


async def wait_jobs(client,headers,sid,timeout=120):
    start=time.perf_counter()
    while time.perf_counter()-start<timeout:
        r=await client.get(f'/sessions/{sid}/jobs',headers=headers);r.raise_for_status();jobs=r.json()
        if any(j['status']=='failed' for j in jobs):raise RuntimeError('Background job failed')
        if jobs and all(j['status']=='done' for j in jobs):return (time.perf_counter()-start)*1000
        await asyncio.sleep(.1)
    raise TimeoutError('Background worker did not finish within timeout')


async def main(args):
    async with httpx.AsyncClient(base_url=args.url,timeout=120,limits=httpx.Limits(max_connections=max(40,args.concurrency+5))) as client:
        capabilities=(await client.get('/system/capabilities')).json()
        headers=await account(client)
        sessions=[]
        for i in range(args.concurrency):
            r=await client.post('/sessions',headers=headers,json=dict(title=f'Load test {i}',subject='DBMS'));r.raise_for_status();sessions.append(r.json()['id'])
        samples=[]
        async def request(sid):
            start=time.perf_counter();first=None;saved=False;generation=None
            try:
                async with client.stream('POST',f'/sessions/{sid}/messages/stream',headers=headers,json={'content':'Explain ACID transactions.'}) as response:
                    response.raise_for_status();kind=''
                    async for line in response.aiter_lines():
                        if line.startswith('event: '):kind=line[7:]
                        if line.startswith('data: '):
                            data=json.loads(line[6:])
                            if kind=='token' and data.get('text') and first is None:first=(time.perf_counter()-start)*1000
                            if kind=='saved':saved=True;generation=data['assistant_message']['generation_ms']
                            if kind=='error':raise RuntimeError(data['detail'])
                if first is None or not saved:raise RuntimeError('Incomplete stream')
                completed=(time.perf_counter()-start)*1000
                checked=await wait_jobs(client,headers,sid)
                return dict(session_id=sid,first_content_ms=round(first,2),stream_complete_ms=round(completed,2),generation_ms=round(generation,2),post_stream_check_wait_ms=round(checked,2),ok=True)
            except Exception as exc:return dict(session_id=sid,ok=False,error=type(exc).__name__+': '+str(exc))
        for _ in range(args.rounds):samples.extend(await asyncio.gather(*(request(sid) for sid in sessions)))
    values=sorted(x['first_content_ms'] for x in samples if x['ok'])
    p95=values[max(0,math.ceil(len(values)*.95)-1)] if values else None
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),url=args.url,capabilities=capabilities,concurrency=args.concurrency,rounds=args.rounds,
        requests=len(samples),successful=len(values),errors=len(samples)-len(values),p95_first_content_ms=p95,median_first_content_ms=statistics.median(values) if values else None,
        under_3s_target=bool(values) and len(values)==len(samples) and p95<3000,
        interpretation='First nonempty SSE content, client measured, nearest-rank P95. Includes network and application time. Offline mode measures orchestration only. This test does not isolate application overhead or validate detection accuracy.',samples=samples)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'samples'}},indent=2))
    print('Saved:',output.resolve())
    if result['errors']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');parser.add_argument('--concurrency',type=int,default=20);parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--output',default='reports/load-test.json');args=parser.parse_args()
    if not 1<=args.concurrency<=100 or not 1<=args.rounds<=100:parser.error('Concurrency and rounds must be between 1 and 100')
    asyncio.run(main(args))
