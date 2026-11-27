import numpy as np
from sklearn.metrics import average_precision_score
from app.ml import X,y,metrics,queue,CAPACITY
from app.data import FEATURES

def test_temporal_split_is_disjoint_and_ordered(experiment):
    tr,va,te=(experiment[k] for k in ('train','validation','test'))
    assert [len(tr),len(va),len(te)]==[5400,1800,1800]
    assert tr[-1]['timestamp'] < va[0]['timestamp'] < va[-1]['timestamp'] < te[0]['timestamp']
    assert not ({r['transaction_id'] for r in tr}&{r['transaction_id'] for r in te})

def test_model_is_selected_on_validation_ap(experiment):
    scores=experiment['comparison']
    assert experiment['model_name']==max(scores,key=lambda m:scores[m]['validation_ap'])
    actual=average_precision_score(y(experiment['validation']),experiment['model'].predict_proba(X(experiment['validation']))[:,1])
    assert abs(actual-scores[experiment['model_name']]['validation_ap'])<1e-12

def test_operating_threshold_is_chosen_on_validation(experiment):
    p=experiment['model'].predict_proba(X(experiment['validation']))[:,1]
    k=int(np.ceil(len(p)*CAPACITY))
    assert experiment['threshold']==np.sort(p)[::-1][k-1]
    assert experiment['validation_metrics']['selected']>=k

def test_feature_set_does_not_include_labels_or_future_data():
    assert 'is_fraud' not in FEATURES and 'timestamp' not in FEATURES

def test_metrics_counts_and_undefined_precision():
    m=metrics([1,0,1,0],[.9,.8,.2,.1],.5)
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(1,1,1,1)
    assert m['precision']==.5 and m['recall']==.5 and m['review_rate']==.5
    m=metrics([1,0],[.3,.2],.9)
    assert m['precision'] is None and m['recall']==0

def test_queue_sort_and_labels(experiment):
    rows=queue(experiment['test'],experiment['test_scores'],experiment['threshold'],20,True)
    assert len(rows)==20 and rows[0]['score']>=rows[-1]['score']
    assert all('is_fraud' in r and r['flagged']==(r['score']>=experiment['threshold']) for r in rows)

def test_reference_perturbation_is_training_based(experiment):
    from app.ml import feature_replacements
    result=feature_replacements(experiment,experiment['test'][0])
    assert len(result['effects'])==len(FEATURES)
    assert abs(result['score']-experiment['test_scores'][0])<1e-12
    assert {e['feature'] for e in result['effects']}==set(FEATURES)
    assert all(abs(e['score_difference'])<=1 for e in result['effects'])
