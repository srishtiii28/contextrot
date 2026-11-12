"""Run the examiner path against a running server; save measured evidence."""
import argparse
import asyncio
import json
from pathlib import Path
import httpx
from load_test import account,wait_jobs


async def main(args):
    async with httpx.AsyncClient(base_url=args.url,timeout=120) as client:
        capabilities=(await client.get('/system/capabilities')).json()
        if not capabilities['demo_faults']:raise RuntimeError('This scripted test requires offline mode with demo faults enabled')
        jwt=await account(client)
        key=await client.post('/api-keys',headers=jwt,json={'name':'Automated demonstration'});key.raise_for_status();key=key.json()
        headers={'X-API-Key':key['key']}
        r=await client.post('/sessions',headers=headers,json={'title':'Examiner path','subject':'DBMS','constraints':['At most 2 sentences.']});r.raise_for_status();sid=r.json()['id']
        checks=[]
        for question,fault in [('Explain primary keys.',None),('Explain photosynthesis.',None),('Explain primary keys.','contradiction'),('Explain primary keys.','instruction'),('Explain primary keys.','persistent')]:
            r=await client.post(f'/sessions/{sid}/messages',headers=headers,json={'content':question,'demo_fault':fault});r.raise_for_status();reply=r.json()
            if reply['job_id']:await wait_jobs(client,headers,sid)
            history=(await client.get(f'/sessions/{sid}/history',headers=headers)).json()
            last=history[-1]
            expected='flagged' if fault=='persistent' else 'corrected' if fault else 'refused' if 'photosynthesis' in question else 'checked'
            assert last['status']==expected,(fault,last)
            checks.append(dict(question=question,injected_fault=fault,expected_status=expected,actual_status=last['status'],passed=True))
        events=(await client.get(f'/sessions/{sid}/events',headers=headers)).json()
        # A new connection must see the same persisted history.
        async with httpx.AsyncClient(base_url=args.url) as reconnected:
            reloaded=(await reconnected.get(f'/sessions/{sid}/history',headers=headers)).json()
            assert reloaded==history
        await client.delete(f"/api-keys/{key['id']}",headers=jwt)
        assert (await client.get('/sessions',headers=headers)).status_code==401
    result=dict(capabilities=capabilities,session_id=sid,checks=checks,reload_persistence=True,revoked_key_rejected=True,history=history,events=events,
        limitation='Scripted fault-injection integration evidence only. Does not establish natural context rot, detector F1, or retrieval quality.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'PASS: sourced answer, refusal, contradiction correction, instruction correction, one-retry limit, persistence and key revocation. Session {sid}.')
    print('Saved:',path.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');parser.add_argument('--output',default='reports/demo-check.json');asyncio.run(main(parser.parse_args()))
