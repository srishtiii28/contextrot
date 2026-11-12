import asyncio
import json
import httpx
import pytest
from app.core.config import get_settings
from app.integrations.rag_client import RetrievedChunk,RetrievalRequest,retrieve
from app.integrations.rot_client import analyze
from app.schemas.analysis import AnalysisInput
from app.services.llm_service import generate_response,compress_history
from app.services.streaming import generate_tokens


def transport(monkeypatch,handler):
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))


def test_gemini_generation_contract(monkeypatch):
    settings=get_settings();monkeypatch.setattr(settings,'llm_provider','gemini');monkeypatch.setattr(settings,'google_api_key','test-secret');monkeypatch.setattr(settings,'gemini_model','test-model')
    def handler(request):
        assert request.url.path.endswith('/test-model:generateContent')
        assert request.headers['x-goog-api-key']=='test-secret'
        prompt=json.loads(json.loads(request.content)['contents'][0]['parts'][0]['text'])
        assert prompt['constraints'][0]['content']=='At most 2 sentences.'
        return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':'Grounded answer [notes, page 1]'}]}}]})
    transport(monkeypatch,handler)
    answer=asyncio.run(generate_response(user_message='keys',conversation_history=[],retrieved_context=[RetrievedChunk(text='key fact',source='notes',page=1)],constraints=[{'content':'At most 2 sentences.'}]))
    assert answer=='Grounded answer [notes, page 1]'


def test_gemini_stream_contract(monkeypatch):
    settings=get_settings();monkeypatch.setattr(settings,'llm_provider','gemini');monkeypatch.setattr(settings,'google_api_key','test-secret');monkeypatch.setattr(settings,'gemini_model','test-model')
    def handler(request):
        assert request.url.path.endswith(':streamGenerateContent') and request.url.params['alt']=='sse'
        frames=[{'candidates':[{'content':{'parts':[{'text':'private reasoning','thought':True},{'text':'Hello '}]}}]}, {'candidates':[{'content':{'parts':[{'text':'world'}]}}]}]
        return httpx.Response(200,text=''.join('data: '+json.dumps(x)+'\n\n' for x in frames))
    transport(monkeypatch,handler)
    async def collect():return [x async for x in generate_tokens('keys',[],[],[])]
    assert asyncio.run(collect())==['Hello ','world']


def test_remote_rag_scope_contract(monkeypatch):
    monkeypatch.setattr(get_settings(),'rag_service_url','https://retriever.example/retrieve')
    transport(monkeypatch,lambda request:httpx.Response(200,json={'in_scope':False,'chunks':[{'text':'irrelevant','source':'notes'}]}))
    assert asyncio.run(retrieve(RetrievalRequest(query='outside',session_id=1))).chunks==[]


def test_remote_detector_rejects_invalid_scores(monkeypatch):
    monkeypatch.setattr(get_settings(),'rot_service_url','https://checker.example/analyze')
    transport(monkeypatch,lambda request:httpx.Response(200,json={'health_score':999}))
    with pytest.raises(ValueError):asyncio.run(analyze(AnalysisInput(session_id=1,current_question='x',current_answer='y')))


def test_history_is_bounded():
    history=[{'role':'user','content':'a'*100},{'role':'assistant','content':'b'*200},{'role':'user','content':'latest'}]
    compact=compress_history(history,150)
    assert sum(len(x['content']) for x in compact)<=150
    assert compact[-1]['content']=='latest'
