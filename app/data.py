"""Synthetic data and strict input validation. No real payment data is bundled."""
import csv
import io
from datetime import datetime, timedelta, timezone
import numpy as np

FEATURES = ('amount', 'hour', 'account_age_days', 'prior_24h_count', 'new_device', 'country_mismatch', 'merchant_category')
NUMERIC = FEATURES[:-1]
CATEGORIES = ('retail', 'travel', 'digital', 'grocery')
CSV_LIMIT_BYTES = 2 * 1024 * 1024
CSV_LIMIT_ROWS = 3000

class DataError(ValueError):
    pass

def generate(n=9000, seed=1729):
    """Simulated, independent records; chronological, reproducible and fictional.

    The outcome mechanism is invented and must not be interpreted as a real
    fraud-generating process or a basis for screening actual people.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        dt = start + timedelta(minutes=20 * i)
        late = i >= n * .8
        category = rng.choice(CATEGORIES, p=[.34,.16,.26,.24] if late else [.39,.12,.22,.27])
        amount = float(np.clip(rng.lognormal(3.9, 1.0), 2, 2500))
        age = int(np.clip(rng.exponential(600), 0, 5000))
        prior = int(rng.poisson(2.3 if late else 1.8))
        device = int(rng.random() < (.13 if late else .10))
        mismatch = int(rng.random() < .09)
        hour = dt.hour
        # Deliberately synthetic probabilities and interactions, not real risk rules.
        logit = (-4.5 + .7*device + .65*mismatch + .7*(amount > 180)
                 + .5*(age < 45) + .5*(prior >= 4) + .38*(hour < 5)
                 + .75*(device and mismatch) + .55*(category == 'digital' and amount > 130))
        probability = 1 / (1 + np.exp(-logit))
        fraud = int(rng.random() < probability)
        records.append({'transaction_id': f'SIM-{i+1:06d}', 'timestamp': dt.isoformat(),
                        'amount': round(amount, 2), 'hour': hour, 'account_age_days': age,
                        'prior_24h_count': prior, 'new_device': device, 'country_mismatch': mismatch,
                        'merchant_category': str(category), 'is_fraud': fraud})
    return records

def parse_rows(data: bytes):
    if len(data) > CSV_LIMIT_BYTES: raise DataError('CSV exceeds 2 MB.')
    try: text = data.decode('utf-8-sig')
    except UnicodeDecodeError as e: raise DataError('CSV must be UTF-8 encoded.') from e
    reader = csv.DictReader(io.StringIO(text, newline=''))
    if not reader.fieldnames: raise DataError('CSV is empty or has no header.')
    fields = [x.strip() for x in reader.fieldnames]
    if len(set(fields)) != len(fields): raise DataError('Duplicate column names are not supported.')
    reader.fieldnames = fields
    missing = set(FEATURES) - set(fields)
    if missing: raise DataError('Missing columns: ' + ', '.join(sorted(missing)))
    rows = []
    for line, raw in enumerate(reader, 2):
        if len(rows) >= CSV_LIMIT_ROWS: raise DataError('CSV exceeds 3,000 rows.')
        if None in raw: raise DataError(f'Row {line}: too many columns.')
        try:
            vals = {k: (raw.get(k) or '').strip() for k in FEATURES}
            amount = float(vals['amount'])
            if not np.isfinite(amount) or not 0 <= amount <= 1_000_000: raise ValueError('amount must be between 0 and 1,000,000')
            ints = {}
            for key, low, high in [('hour',0,23), ('account_age_days',0,36500), ('prior_24h_count',0,1000), ('new_device',0,1), ('country_mismatch',0,1)]:
                if not vals[key].isdigit(): raise ValueError(f'{key} must be an integer from {low} to {high}')
                ints[key] = int(vals[key])
                if not low <= ints[key] <= high: raise ValueError(f'{key} must be an integer from {low} to {high}')
            category=vals['merchant_category'].lower()
            if category not in CATEGORIES: raise ValueError('merchant_category must be retail, travel, digital or grocery')
            ident=(raw.get('transaction_id') or f'ROW-{line-1}').strip()[:80]
            if not ident: ident=f'ROW-{line-1}'
            rows.append({'transaction_id':ident, 'amount':amount, **ints, 'merchant_category':category})
        except (ValueError, TypeError) as e: raise DataError(f'Row {line}: {e}') from e
    if not rows: raise DataError('CSV has no data rows.')
    return rows
