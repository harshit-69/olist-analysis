import duckdb, pandas as pd
from scipy import stats

con = duckdb.connect('olist.duckdb')
con.execute(open('sql/01_master.sql').read())
df = con.execute("SELECT * FROM master").df()

# ---- Delivery delay ----
df['delay_days'] = (df.delivered_ts - df.estimated_ts).dt.total_seconds() / 86400
df['late'] = df.delay_days > 0
print("Late %:", round(df.late.mean() * 100, 1))
print(df.groupby('late').review_score.mean())

a = df[df.late].review_score.dropna()
b = df[~df.late].review_score.dropna()
print("Mann-Whitney U:", stats.mannwhitneyu(a, b))

late_by_state = (df.groupby('customer_state')
                   .agg(orders=('order_id', 'count'), late_pct=('late', 'mean'))
                   .query('orders > 300').sort_values('late_pct', ascending=False))
print(late_by_state.head(5))

# ---- Repeat purchase rate ----
cust = df.groupby('customer_unique_id').agg(
    orders=('order_id', 'nunique'), revenue=('revenue', 'sum'),
    last=('purchase_ts', 'max'))
print("Repeat rate %:", round((cust.orders > 1).mean() * 100, 2))

# Does a late first order reduce repeat purchase?
first = df.sort_values('purchase_ts').groupby('customer_unique_id').first()
first['repeat'] = first.index.map(cust.orders > 1)
print(first.groupby('late').repeat.mean())

# ---- Cohort retention ----
df['month'] = df.purchase_ts.dt.to_period('M')
df['cohort'] = df.groupby('customer_unique_id').purchase_ts.transform('min').dt.to_period('M')
df['idx'] = (df.month - df.cohort).apply(lambda x: x.n)
cohort = df.pivot_table(index='cohort', columns='idx',
                        values='customer_unique_id', aggfunc='nunique')
retention = cohort.div(cohort[0], axis=0)

# ---- RFM segmentation ----
snap = df.purchase_ts.max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_unique_id').agg(
    R=('purchase_ts', lambda s: (snap - s.max()).days),
    F=('order_id', 'nunique'), M=('revenue', 'sum'))
rfm['R_s'] = pd.qcut(rfm.R, 4, labels=[4, 3, 2, 1]).astype(int)
rfm['M_s'] = pd.qcut(rfm.M, 4, labels=[1, 2, 3, 4]).astype(int)

def seg(r):
    if r.F > 1 and r.R_s >= 3: return 'Loyal'
    if r.R_s == 4 and r.M_s >= 3: return 'High-value new'
    if r.R_s <= 2 and r.M_s == 4: return 'At-risk big spenders'
    return 'Low value'

rfm['segment'] = rfm.apply(seg, axis=1)
print(rfm.segment.value_counts())

# ---- Save for dashboard ----
df.to_parquet('app/orders.parquet')
retention.to_csv('app/retention.csv')
rfm.to_parquet('app/rfm.parquet')