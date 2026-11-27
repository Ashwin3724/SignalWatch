from pathlib import Path
from app.ml import y

EXAMPLE=Path(__file__).resolve().parent.parent/'examples'/'fictional_transactions.csv'

def test_home_and_sample(client):
    assert client.get('/').status_code==200
    assert 'SignalWatch' in client.get('/').text
    assert client.get('/sample.csv').status_code==200

def test_report_and_test_accounting(client,experiment):
    r=client.get('/api/report');assert r.status_code==200
    d=r.json();assert d['model']==experiment['model_name'] and d['split']['test']['rows']==1800
    m=d['test_metrics'];assert m['tp']+m['fp']+m['fn']+m['tn']==m['total']==1800
    assert m['positive']==sum(y(experiment['test']))

def test_holdout_threshold_changes_workload(client):
    lo=client.get('/api/holdout?threshold=0&limit=3').json()
    hi=client.get('/api/holdout?threshold=1&limit=3').json()
    assert lo['metrics']['selected']==1800
    assert hi['metrics']['selected']<=lo['metrics']['selected']
    assert len(lo['rows'])==3 and all('is_fraud' in x for x in lo['rows'])

def test_invalid_thresholds(client):
    assert client.get('/api/holdout?threshold=1.1').status_code==422
    assert client.get('/api/holdout?threshold=-1').status_code==422

def test_upload_score_does_not_return_labels(client):
    r=client.post('/api/score?threshold=0.5',files={'file':('demo.csv',EXAMPLE.read_bytes(),'text/csv')})
    assert r.status_code==200,r.text
    d=r.json();assert d['total']==5 and len(d['rows'])==5
    assert all('is_fraud' not in x for x in d['rows'])

def test_bad_csv_and_extension(client):
    assert client.post('/api/score?threshold=0.5',files={'file':('a.txt',b'bad')}).status_code==400
    assert client.post('/api/score?threshold=0.5',files={'file':('a.csv',b'amount\n10\n')}).status_code==400

def test_host_and_origin_guards(client):
    assert client.get('/api/report',headers={'Host':'attacker.test'}).status_code==403
    assert client.get('/api/report',headers={'Origin':'https://attacker.test'}).status_code==403
    assert client.get('/api/report',headers={'Origin':'http://127.0.0.1:8010'}).status_code==200

def test_csv_content_is_returned_as_data(client):
    raw=EXAMPLE.read_text().replace('DEMO-001','<img src=x onerror=alert(1)>').encode()
    result=client.post('/api/score?threshold=0.5',files={'file':('x.csv',raw,'text/csv')}).json()
    assert any('<img' in r['transaction_id'] for r in result['rows'])
    # Browser test verifies textContent rendering; API preserves the literal ID.

def test_explanation_endpoint(client,experiment):
    row=experiment['test'][0]
    r=client.post('/api/explain',json=row)
    assert r.status_code==200 and len(r.json()['effects'])==7
    assert client.post('/api/explain',json={**row,'hour':90}).status_code==422
