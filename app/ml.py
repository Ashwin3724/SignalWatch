"""Time-ordered model selection and evaluation on explicitly synthetic data."""
from math import ceil
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from app.data import FEATURES, NUMERIC

SEED=1729
CAPACITY=0.05

def X(rows):
    return [[r[f] for f in FEATURES] for r in rows]
def y(rows):
    return np.asarray([r['is_fraud'] for r in rows],dtype=int)
def pipeline(classifier):
    preprocess=ColumnTransformer([
        ('num',StandardScaler(), list(range(len(NUMERIC)))),
        ('category',OneHotEncoder(handle_unknown='ignore'),[len(NUMERIC)])])
    return Pipeline([('preprocess',preprocess),('classifier',classifier)])

def train(records):
    """Strict time split: 60% train, 20% validation, 20% untouched test.

    Preprocessors fit on training only. Model selected on validation AP. The
    operating threshold targets 5% review on validation, not on the test set.
    """
    records=sorted(records,key=lambda x:x['timestamp'])
    n=len(records); a=int(n*.6); b=int(n*.8)
    tr,va,te=records[:a],records[a:b],records[b:]
    assert len(tr)>0 and len(va)>0 and len(te)>0
    if set(y(tr)) != {0,1} or set(y(va)) != {0,1} or set(y(te)) != {0,1}:
        raise ValueError('Each split must contain both synthetic classes.')
    models={
        'Logistic regression':pipeline(LogisticRegression(max_iter=1000,random_state=SEED)),
        'Random forest':pipeline(RandomForestClassifier(n_estimators=100,min_samples_leaf=6,max_depth=12,n_jobs=-1,random_state=SEED))}
    stats={}
    for name,model in models.items():
        model.fit(X(tr),y(tr))
        p=model.predict_proba(X(va))[:,1]
        stats[name]={'validation_ap':float(average_precision_score(y(va),p)),
                     'validation_brier':float(brier_score_loss(y(va),p))}
    chosen=max(models,key=lambda name:stats[name]['validation_ap'])
    model=models[chosen]
    vp=model.predict_proba(X(va))[:,1]
    k=max(1,ceil(len(va)*CAPACITY))
    # Threshold includes ties, so observed rate can exceed the target.
    threshold=float(np.sort(vp)[::-1][k-1])
    test_p=model.predict_proba(X(te))[:,1]
    val_metrics=metrics(y(va),vp,threshold)
    test_metrics=metrics(y(te),test_p,threshold)
    return {'model':model,'model_name':chosen,'comparison':stats,'threshold':threshold,
            'train':tr,'validation':va,'test':te,'test_scores':test_p,
            'validation_metrics':val_metrics,'test_metrics':test_metrics}

def metrics(labels,scores,threshold):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    flag=scores >= threshold
    tp=int(np.sum(flag & (labels==1))); fp=int(np.sum(flag & (labels==0)))
    fn=int(np.sum(~flag & (labels==1))); tn=int(np.sum(~flag & (labels==0)))
    selected=tp+fp; positives=tp+fn; total=len(labels)
    return {'total':total,'positive':positives,'selected':selected,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/selected if selected else None,
            'recall':tp/positives if positives else None,
            'review_rate':selected/total if total else None,
            'average_precision':float(average_precision_score(labels,scores)) if positives else None,
            'brier':float(brier_score_loss(labels,scores)) if total else None}

def queue(rows,scores,threshold,limit=100,include_labels=False):
    order=sorted(range(len(rows)), key=lambda i:(-float(scores[i]),str(rows[i]['transaction_id'])))
    output=[]
    for i in order[:limit]:
        r=rows[i]
        item={k:r[k] for k in ('transaction_id',*FEATURES)}
        if 'timestamp' in r:item['timestamp']=r['timestamp']
        if include_labels:item['is_fraud']=r['is_fraud']
        item['score']=float(scores[i]);item['flagged']=bool(scores[i]>=threshold)
        output.append(item)
    return output

def feature_replacements(experiment,row):
    """Describe score sensitivity to one-at-a-time training-reference changes.

    Not a causal attribution, SHAP value, or explanation of real fraud.
    The reference is computed strictly from the training period.
    """
    from collections import Counter
    tr=experiment['train']
    references={f:float(np.median([r[f] for r in tr])) for f in NUMERIC}
    references['merchant_category']=Counter(r['merchant_category'] for r in tr).most_common(1)[0][0]
    variants=[]
    for feature in FEATURES:
        altered={**row,feature:references[feature]}
        variants.append(altered)
    model=experiment['model']
    original=float(model.predict_proba(X([row]))[0,1])
    changed=model.predict_proba(X(variants))[:,1]
    effects=[{'feature':f,'value':row[f],'reference':references[f],
              'score_difference':original-float(p)} for f,p in zip(FEATURES,changed)]
    effects.sort(key=lambda item:-abs(item['score_difference']))
    return {'score':original,'effects':effects,'reference':'Training-set median for numeric features; training-set mode for category.'}
