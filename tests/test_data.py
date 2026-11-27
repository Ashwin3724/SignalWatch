import pytest
from app.data import generate,parse_rows,DataError,FEATURES

def test_synthetic_generation_reproducible():
    assert generate(12)==generate(12)
    assert generate(12)!=generate(12,3)
    assert all(r['transaction_id'].startswith('SIM-') for r in generate(5))

def test_example_parses_and_no_label_as_feature():
    data=b'transaction_id,amount,hour,account_age_days,prior_24h_count,new_device,country_mismatch,merchant_category,is_fraud\nS1,22.2,12,30,1,0,1,digital,1\n'
    rows=parse_rows(data)
    assert len(rows)==1 and rows[0]['amount']==22.2 and 'is_fraud' not in rows[0]
    assert 'is_fraud' not in FEATURES

@pytest.mark.parametrize('replacement', ['nan','inf','-1','1000001','not-a-number'])
def test_bad_amounts(replacement):
    row=f'amount,hour,account_age_days,prior_24h_count,new_device,country_mismatch,merchant_category\n{replacement},12,3,1,1,0,retail\n'.encode()
    with pytest.raises(DataError):parse_rows(row)

@pytest.mark.parametrize('row', ['1,24,3,1,1,0,retail','1,12,3,1,2,0,retail','1,12,3,1,1,0,unknown','1,12,3,1,1,0,retail,extra'])
def test_bad_columns(row):
    data=('amount,hour,account_age_days,prior_24h_count,new_device,country_mismatch,merchant_category\n'+row+'\n').encode()
    with pytest.raises(DataError):parse_rows(data)

def test_missing_columns():
    with pytest.raises(DataError,match='Missing columns'):parse_rows(b'amount,hour\n2,1\n')

def test_bad_utf8():
    with pytest.raises(DataError,match='UTF-8'):parse_rows(b'\xff')

def test_duplicate_headers():
    with pytest.raises(DataError,match='Duplicate'):parse_rows(b'amount,amount\n1,2\n')

def test_size_limit():
    with pytest.raises(DataError,match='2 MB'):parse_rows(b'x'*(2*1024*1024+1))
