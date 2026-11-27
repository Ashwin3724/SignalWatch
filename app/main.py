"""Local-only educational fraud review demo; no persistence of user CSVs."""
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
import logging
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from app.data import generate, parse_rows, DataError, CSV_LIMIT_BYTES
from app.ml import train, X, y, metrics, queue, CAPACITY, feature_replacements

ROOT=Path(__file__).resolve().parent.parent
logger=logging.getLogger(__name__)

class Features(BaseModel):
    amount: float = Field(ge=0,le=1_000_000,allow_inf_nan=False)
    hour: int = Field(ge=0,le=23)
    account_age_days: int = Field(ge=0,le=36500)
    prior_24h_count: int = Field(ge=0,le=1000)
    new_device: int = Field(ge=0,le=1)
    country_mismatch: int = Field(ge=0,le=1)
    merchant_category: Literal['retail','travel','digital','grocery']

def create_app(state=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.experiment is None:
            app.state.experiment=train(generate())
        yield
    app=FastAPI(title='SignalWatch',version='1.0.0',lifespan=lifespan)
    app.mount('/static',StaticFiles(directory=ROOT/'app'/'static'),name='static')
    app.state.experiment=state
    @app.middleware('http')
    async def local_only(request:Request,call_next):
        if request.url.hostname not in ('127.0.0.1','localhost','::1'):
            return JSONResponse({'detail':'This is a local-only app.'},status_code=403)
        origin=request.headers.get('origin')
        if origin:
            try:
                parsed=urlsplit(origin)
                op=parsed.port or (443 if parsed.scheme=='https' else 80)
                tp=request.url.port or (443 if request.url.scheme=='https' else 80)
                if parsed.scheme!=request.url.scheme or parsed.hostname!=request.url.hostname or op!=tp:
                    return JSONResponse({'detail':'Cross-origin requests are not allowed.'},status_code=403)
            except ValueError:return JSONResponse({'detail':'Invalid origin.'},status_code=403)
        res=await call_next(request)
        res.headers['X-Content-Type-Options']='nosniff'
        res.headers['Referrer-Policy']='no-referrer'
        if request.url.path not in ('/docs','/openapi.json'):
            res.headers['Content-Security-Policy']="default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        return res
    @app.get('/',include_in_schema=False)
    def index():return FileResponse(ROOT/'app'/'static'/'index.html')
    @app.get('/sample.csv',include_in_schema=False)
    def sample_csv():return FileResponse(ROOT/'examples'/'fictional_transactions.csv',media_type='text/csv',filename='fictional_transactions.csv')
    @app.get('/api/report')
    def report():
        e=app.state.experiment
        if e is None:raise HTTPException(503,'Model is still starting.')
        return {'dataset':'Synthetic transactions generated locally; not real financial data.',
                'model':e['model_name'],'comparison':e['comparison'],'threshold':e['threshold'],
                'review_target':CAPACITY,'validation_metrics':e['validation_metrics'],
                'test_metrics':e['test_metrics'],
                'split':{k:{'rows':len(e[k]),'start':e[k][0]['timestamp'],'end':e[k][-1]['timestamp']} for k in ('train','validation','test')}}
    @app.get('/api/holdout')
    def holdout(threshold:float=Query(ge=0,le=1), limit:int=Query(default=100,ge=1,le=200)):
        e=app.state.experiment
        if e is None:raise HTTPException(503,'Model is still starting.')
        return {'metrics':metrics(y(e['test']),e['test_scores'],threshold),
                'rows':queue(e['test'],e['test_scores'],threshold,limit,True),'shown_limit':limit}
    @app.post('/api/explain')
    def explain(body:Features):
        e=app.state.experiment
        if e is None:raise HTTPException(503,'Model is still starting.')
        return feature_replacements(e,body.model_dump())
    @app.post('/api/score')
    async def score(file:UploadFile=File(...),threshold:float=Query(ge=0,le=1)):
        e=app.state.experiment
        if e is None:raise HTTPException(503,'Model is still starting.')
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(400,'Choose a .csv file.')
        data=await file.read(CSV_LIMIT_BYTES+1);await file.close()
        try:rows=parse_rows(data)
        except DataError as exc:raise HTTPException(400,str(exc)) from exc
        probs=e['model'].predict_proba(X(rows))[:,1]
        return {'total':len(rows),'flagged':int(sum(p>=threshold for p in probs)),
                'rows':queue(rows,probs,threshold,25,False), 'shown_limit':25}
    return app

app=create_app()
