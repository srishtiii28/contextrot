import asyncio
import time
from sqlalchemy import select, update
from app.database.database import SessionLocal
from app.database.runtime_models import ApiKey, Job
from app.services.worker import process_one, claim_job


def session(client,headers,constraints=None):
    r=client.post('/sessions',headers=headers,json=dict(title='DBMS revision',subject='DBMS',constraints=constraints or []))
    assert r.status_code==201,r.text
    return r.json()['id']


def send(client,headers,sid,fault=None):
    r=client.post(f'/sessions/{sid}/messages',headers=headers,json=dict(content='Explain primary keys.',demo_fault=fault))
    assert r.status_code==201,r.text
    return r.json()


def drain():
    while asyncio.run(process_one()):pass


def test_background_correction_and_reload(client,account):
    sid=session(client,account,['At most 2 sentences.'])
    reply=send(client,account,sid,'contradiction')
    assert reply['assistant_message']['status']=='pending'
    assert client.get(f'/sessions/{sid}/analysis',headers=account).json()==[]
    drain()
    history=client.get(f'/sessions/{sid}/history',headers=account).json()
    assert len(history)==3
    assert history[1]['status']=='superseded'
    assert history[2]['correction_of']==reply['assistant_message']['id']
    assert history[2]['status']=='corrected'
    assert history[2]['sources']
    assert 'cannot contain NULL' in history[2]['content']
    # A fresh client/lifespan reads the same persisted correction.
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as other:
        assert other.get(f'/sessions/{sid}/history',headers=account).json()==history
    assert client.get(f'/sessions/{sid}/health',headers=account).json()['current_score']==100


def test_persistent_failure_stops_after_one_retry(client,account):
    sid=session(client,account)
    send(client,account,sid,'persistent');drain();drain()
    history=client.get(f'/sessions/{sid}/history',headers=account).json()
    assert len(history)==3 and history[-1]['status']=='flagged'
    assert client.get(f'/sessions/{sid}/jobs',headers=account).json()[0]['status']=='done'


def test_instruction_and_drift(client,account):
    for fault in ['instruction','drift']:
        sid=session(client,account,['At most 2 sentences.'])
        send(client,account,sid,fault);drain()
        records=client.get(f'/sessions/{sid}/analysis',headers=account).json()
        assert records[0]['memory_violation' if fault=='instruction' else 'topic_drift']
        assert records[-1]['health_score']==100


def test_refusal_and_clean_answer(client,account):
    sid=session(client,account)
    refused=client.post(f'/sessions/{sid}/messages',headers=account,json={'content':'Explain photosynthesis.'}).json()
    assert refused['assistant_message']['status']=='refused' and refused['job_id'] is None
    send(client,account,sid);drain()
    assert len(client.get(f'/sessions/{sid}/history',headers=account).json())==4
    assert client.get(f'/sessions/{sid}/health',headers=account).json()['current_score']==100


def test_key_hash_revoke_and_session_isolation(client,account):
    sid=session(client,account)
    key=client.post('/api-keys',headers=account,json={'name':'test'}).json()
    headers={'X-API-Key':key['key']}
    assert client.get(f'/sessions/{sid}',headers=headers).status_code==200
    with SessionLocal() as db:assert db.get(ApiKey,key['id']).digest!=key['key']
    assert client.get(f'/sessions/{sid}').status_code==401
    client.post('/auth/signup',json={'username':'other_user','email':'other@example.com','password':'password-123'})
    token=client.post('/auth/login',json={'username':'other_user','password':'password-123'}).json()['access_token']
    other={'Authorization':'Bearer '+token}
    assert client.get(f'/sessions/{sid}/history',headers=other).status_code==404
    assert client.post('/integrations/rag/retrieve',headers=other,json={'session_id':sid,'query':'keys'}).status_code==404
    client.delete(f"/api-keys/{key['id']}",headers=account)
    assert client.get('/sessions',headers=headers).status_code==401


def test_websocket_replays_persisted_events(client,account):
    sid=session(client,account);send(client,account,sid,'contradiction');drain()
    events=client.get(f'/sessions/{sid}/events',headers=account).json()
    assert [e['kind'] for e in events]==['answer.saved','analysis.completed','answer.corrected']
    with client.websocket_connect(f'/sessions/{sid}/events?after={events[0]["id"]}') as ws:
        ws.send_json({'token':account['Authorization'][7:]})
        assert ws.receive_json()['kind']=='connected'
        assert ws.receive_json()['id']==events[1]['id']
        assert ws.receive_json()['id']==events[2]['id']


def test_expired_lease_reclaimed_without_duplicate_correction(client,account):
    sid=session(client,account);send(client,account,sid,'contradiction')
    claimed=claim_job();assert claimed
    with SessionLocal() as db:
        db.execute(update(Job).where(Job.id==claimed[0]).values(lease_until=time.time()-1));db.commit()
    drain();drain()
    assert len(client.get(f'/sessions/{sid}/history',headers=account).json())==3


def test_failed_detector_is_unverified_then_manual_retry(client,account,monkeypatch):
    import app.services.worker as worker
    original=worker.analyze
    async def broken(_):raise RuntimeError('service down')
    monkeypatch.setattr(worker,'analyze',broken)
    sid=session(client,account);reply=send(client,account,sid);drain()
    assert client.get(f'/sessions/{sid}/history',headers=account).json()[-1]['status']=='check_failed'
    monkeypatch.setattr(worker,'analyze',original)
    assert client.post(f'/sessions/{sid}/jobs/{reply["job_id"]}/retry',headers=account).status_code==200
    drain()
    assert client.get(f'/sessions/{sid}/history',headers=account).json()[-1]['status']=='checked'


def test_stream_and_pending_guard(client,account):
    sid=session(client,account)
    r=client.post(f'/sessions/{sid}/messages/stream',headers=account,json={'content':'Explain SQL joins.'})
    assert 'event: token' in r.text and 'event: saved' in r.text
    assert client.post(f'/sessions/{sid}/messages',headers=account,json={'content':'Explain keys.'}).status_code==409
    drain()


def test_demo_fault_gate_and_invalid_provider(client,account,monkeypatch):
    from app.core.config import get_settings
    s=get_settings();sid=session(client,account)
    monkeypatch.setattr(s,'enable_demo_faults',False)
    assert client.post(f'/sessions/{sid}/messages',headers=account,json={'content':'Explain primary keys.','demo_fault':'contradiction'}).status_code==403
    monkeypatch.setattr(s,'llm_provider','invalid')
    assert client.post(f'/sessions/{sid}/messages',headers=account,json={'content':'Explain primary keys.'}).status_code==503
    assert client.get(f'/sessions/{sid}/history',headers=account).json()==[]


def test_overlapping_generation_is_serialized(client,account,monkeypatch):
    import app.routes.chat as chat
    from concurrent.futures import ThreadPoolExecutor
    original=chat.generate_response
    async def slow(**kwargs):
        await asyncio.sleep(.3)
        return await original(**kwargs)
    monkeypatch.setattr(chat,'generate_response',slow)
    sid=session(client,account)
    def submit():return client.post(f'/sessions/{sid}/messages',headers=account,json={'content':'Explain primary keys.'}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(submit) for _ in range(2)]
        assert sorted(t.result() for t in tasks)==[201,409]
    drain()
    assert len(client.get(f'/sessions/{sid}/history',headers=account).json())==2


def test_submission_lease_expiry(client,account):
    from app.services.submission_lock import acquire,release
    from app.database.runtime_models import SessionLease
    from fastapi import HTTPException
    import pytest
    sid=session(client,account);old=acquire(sid)
    with pytest.raises(HTTPException):acquire(sid)
    with SessionLocal() as db:
        db.execute(update(SessionLease).where(SessionLease.session_id==sid).values(expires_at=time.time()-1));db.commit()
    fresh=acquire(sid);release(sid,old)  # A stale owner cannot release the new lease.
    with pytest.raises(HTTPException):acquire(sid)
    release(sid,fresh)
