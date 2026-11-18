from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu

# Palette: ink (text), harbor (on-time / primary), signal (late), mist (surfaces), slate (secondary text)
INK, HARBOR, SIGNAL, MIST, SLATE, RULE = "#16233B", "#1F6F8B", "#E4572E", "#EEF3F6", "#6B7A90", "#D9E2E8"
REPO = "https://github.com/harshit-69/olist-analysis"
DATA = Path(__file__).parent / "orders.parquet"

st.set_page_config(page_title="Olist retention & delivery analysis", page_icon="📦", layout="wide")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stCaptionContainer"], .stTabs [data-baseweb="tab"] p {{ font-family: 'Manrope', sans-serif; }}
.block-container {{ max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}
.hero h1 {{ font-size: 2.2rem; line-height: 1.2; font-weight: 800; letter-spacing: -0.02em;
  color: {INK}; max-width: 26em; margin: 0 0 .7rem; padding: 0; }}
.hero p {{ color: {SLATE}; font-size: 1.05rem; line-height: 1.55; max-width: 60ch; margin: 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid {RULE};
  border-bottom: 1px solid {RULE}; margin: 1.8rem 0 1.4rem; }}
.kpi {{ padding: 1.1rem 1.3rem; border-left: 1px solid {RULE}; }}
.kpi:first-child {{ border-left: 0; padding-left: 0; }}
.kpi b {{ display: block; font-size: 1.95rem; font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
.kpi span {{ color: {SLATE}; font-size: .92rem; }}
.note {{ color: {SLATE}; font-size: .95rem; line-height: 1.55; max-width: 66ch; margin: .2rem 0 1.4rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .7rem 0; }}
h3 {{ font-weight: 700 !important; letter-spacing: -0.01em; }}
@media (max-width: 760px) {{
  .kpis {{ grid-template-columns: repeat(2, 1fr); }}
  .kpi:nth-child(3) {{ border-left: 0; padding-left: 0; }}
  .kpi {{ border-bottom: 1px solid {RULE}; }}
  .hero h1 {{ font-size: 1.7rem; }}
}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load() -> pd.DataFrame:
    d = pd.read_parquet(DATA)
    d["purchase_ts"] = pd.to_datetime(d["purchase_ts"])
    if "delay_days" not in d:
        d["delay_days"] = (pd.to_datetime(d["delivered_ts"]) - pd.to_datetime(d["estimated_ts"])).dt.total_seconds() / 86400
    d["late"] = d["delay_days"] > 0
    return d[["order_id", "customer_unique_id", "customer_state", "purchase_ts", "revenue",
              "review_score", "delay_days", "late"]]


def fmt(v, spec=".1f", suffix=""):
    return "n/a" if v is None or pd.isna(v) else f"{v:{spec}}{suffix}"


def money(v):
    return f"R$ {v / 1e6:.1f}M" if v >= 1e6 else f"R$ {v / 1e3:.0f}K"


def tidy(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope, sans-serif", color=INK, size=13),
                      hoverlabel=dict(font_family="Manrope, sans-serif"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=RULE)
    fig.update_yaxes(gridcolor="#E6ECF0", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, theme=None)


def retention_matrix(d, months=6, min_size=50):
    t = d[["customer_unique_id", "purchase_ts"]].copy()
    t["m"] = t.purchase_ts.dt.year * 12 + t.purchase_ts.dt.month
    t["first"] = t.groupby("customer_unique_id").m.transform("min")
    t["idx"] = t.m - t["first"]
    last_m = t.m.max()
    cnt = (t[t.idx <= months].groupby(["first", "idx"]).customer_unique_id.nunique()
           .unstack(fill_value=0).reindex(columns=range(months + 1), fill_value=0))
    size = cnt[0]
    pct = cnt.div(size, axis=0) * 100
    for k in pct.columns:
        pct.loc[np.asarray(last_m - pct.index) < k, k] = np.nan
    keep = (size >= min_size).values
    pct, size = pct[keep], size[keep]
    labels = [f"{pd.Timestamp(year=(f - 1) // 12, month=(f - 1) % 12 + 1, day=1):%b %Y}" for f in pct.index]
    return pct.drop(columns=0), size, labels


SEG_ORDER = ["Repeat buyers", "Recent big spenders", "Lapsed big spenders", "Everyone else"]


def segments(d):
    g = d.groupby("customer_unique_id").agg(last=("purchase_ts", "max"), F=("order_id", "nunique"), M=("revenue", "sum"))
    if len(g) < 8:
        return None
    g["R"] = (d.purchase_ts.max() + pd.Timedelta(days=1) - g["last"]).dt.days
    g["Rs"] = pd.qcut(g.R.rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    g["Ms"] = pd.qcut(g.M.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["segment"] = np.select(
        [(g.F > 1) & (g.Rs >= 3), (g.Rs == 4) & (g.Ms >= 3), (g.Rs <= 2) & (g.Ms == 4)],
        SEG_ORDER[:3], SEG_ORDER[3])
    out = g.groupby("segment").agg(customers=("M", "size"), revenue=("M", "sum"))
    out["share"] = out.revenue / out.revenue.sum() * 100
    return out.reindex(SEG_ORDER).dropna()


# ---------- data + filters ----------
data = load()
with st.sidebar:
    st.markdown("### Filters")
    states = st.multiselect("State", sorted(data.customer_state.unique()), placeholder="All states")
    lo, hi = data.purchase_ts.min().date(), data.purchase_ts.max().date()
    rng = st.date_input("Order date", (lo, hi), min_value=lo, max_value=hi)
    st.caption("Filters apply to every chart on the page.")

df = data
if states:
    df = df[df.customer_state.isin(states)]
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    df = df[(df.purchase_ts.dt.date >= rng[0]) & (df.purchase_ts.dt.date <= rng[1])]
if df.empty:
    st.warning("No orders match these filters. Widen the date range or clear the state selection.")
    st.stop()

# ---------- headline numbers ----------
on_score = df.loc[~df.late, "review_score"].mean()
late_score = df.loc[df.late, "review_score"].mean()
orders_per = df.groupby("customer_unique_id").order_id.nunique()
repeat_pct = (orders_per > 1).mean() * 100

if pd.notna(on_score) and pd.notna(late_score):
    headline = (f"Late orders average {late_score:.1f} stars against {on_score:.1f} for on-time ones, "
                f"and only {repeat_pct:.1f}% of customers ever buy again.")
else:
    headline = f"Only {repeat_pct:.1f}% of customers ever buy again."

st.markdown(f"""<div class="hero"><h1>{headline}</h1>
<p>{len(df):,} delivered orders from Olist, a Brazilian marketplace, placed between
{df.purchase_ts.min():%b %Y} and {df.purchase_ts.max():%b %Y}. Pick a state or date range on the left to narrow it down.</p></div>
<div class="kpis">
<div class="kpi"><b>{money(df.revenue.sum())}</b><span>Revenue</span></div>
<div class="kpi"><b>{df.customer_unique_id.nunique():,}</b><span>Customers</span></div>
<div class="kpi"><b>{df.late.mean() * 100:.1f}%</b><span>Orders delivered late</span></div>
<div class="kpi"><b>{fmt(df.review_score.mean(), ".2f")}</b><span>Average review, out of 5</span></div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Delivery and reviews", "Who comes back", "Customer value", "About this project"])

# ---------- tab 1 ----------
with tab1:
    st.markdown("### Review scores by delivery timing")
    labels = ["A week or more early", "Up to a week early", "1 to 3 days late", "4 to 7 days late", "8+ days late"]
    colors = ["#7FB2C5", HARBOR, "#F2A488", SIGNAL, "#B3341A"]
    b = df.assign(bucket=pd.cut(df.delay_days, [-np.inf, -7, 0, 3, 7, np.inf], labels=labels)) \
          .groupby("bucket", observed=False).agg(score=("review_score", "mean"), orders=("order_id", "count"))
    fig = go.Figure(go.Bar(x=labels, y=b.score.fillna(0), marker_color=colors, customdata=b.orders,
                           text=[fmt(v, ".2f") for v in b.score], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>Average review %{y:.2f}<br>%{customdata:,} orders<extra></extra>"))
    fig.update_yaxes(range=[0, 5.4], tickvals=[0, 1, 2, 3, 4, 5])
    show(tidy(fig, 340))

    a, c = df[df.late].review_score.dropna(), df[~df.late].review_score.dropna()
    first = df.sort_values("purchase_ts").drop_duplicates("customer_unique_id")
    rep = first.assign(rep=first.customer_unique_id.map(orders_per > 1)).groupby("late").rep.mean() * 100
    note = ""
    if len(a) > 20 and len(c) > 20:
        p = mannwhitneyu(a, c).pvalue
        note += f"The gap between late and on-time review scores is statistically significant (Mann-Whitney U test, {'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}). "
    if False in rep.index and True in rep.index:
        note += (f"{rep[False]:.1f}% of customers whose first order arrived on time ordered again, "
                 f"against {rep[True]:.1f}% of those whose first order was late.")
    if note:
        st.markdown(f"<p class='note'>{note}</p>", unsafe_allow_html=True)

    st.markdown("### States with the most late deliveries")
    s = data.groupby("customer_state").agg(orders=("order_id", "count"), late=("late", "mean"))
    s = s[s.orders >= 100]
    s = s[s.index.isin(states)] if states else s
    national = data.late.mean() * 100
    if len(s) >= 2:
        top = s.sort_values("late", ascending=False).head(8).iloc[::-1]
        fig = go.Figure(go.Bar(y=top.index, x=top.late * 100, orientation="h", marker_color=SIGNAL,
                               text=[f"{v:.1f}%" for v in top.late * 100], textposition="outside", cliponaxis=False,
                               customdata=top.orders, hovertemplate="%{y}: %{x:.1f}% late<br>%{customdata:,} orders<extra></extra>"))
        fig.add_vline(x=national, line_dash="dot", line_color=SLATE,
                      annotation_text=f"All states {national:.1f}%", annotation_position="bottom right",
                      annotation_font_color=SLATE)
        fig.update_xaxes(ticksuffix="%", range=[0, top.late.max() * 100 * 1.18])
        fig.update_yaxes(showgrid=False)
        show(tidy(fig, 330))
        worst5 = s.sort_values("late", ascending=False).head(5)
        avoid = (np.clip(worst5.late * 100 - national, 0, None) / 100 * worst5.orders).sum()
        st.markdown(f"<p class='note'>If the five slowest states delivered late as rarely as the all-state average, "
                    f"roughly {int(round(avoid, -1)):,} fewer orders would arrive late.</p>", unsafe_allow_html=True)
    else:
        st.info("Select more than one state, or clear the state filter, to compare states.")

# ---------- tab 2 ----------
with tab2:
    st.markdown("### Share of each first-month cohort that orders again")
    st.markdown("<p class='note'>Each row is the group of customers who first ordered in that month. "
                "Each column shows the share of that group who ordered again that many months later. "
                "Blank cells are months the data does not reach.</p>", unsafe_allow_html=True)
    pct, size, rows = retention_matrix(df)
    if len(pct) >= 2:
        z = pct.values
        text = np.where(np.isnan(z), "", np.char.add(np.round(z, 1).astype(str), "%"))
        fig = go.Figure(go.Heatmap(z=z, x=[f"Month {k}" for k in pct.columns], y=[f"{r}  ({n:,})" for r, n in zip(rows, size)],
                                   text=text, texttemplate="%{text}", xgap=3, ygap=3, showscale=False,
                                   colorscale=[[0, MIST], [1, HARBOR]], zmin=0, zmax=max(1.0, np.nanmax(z)),
                                   hovertemplate="%{y}<br>%{x}: %{z:.2f}% ordered again<extra></extra>"))
        fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
        fig.update_xaxes(side="top")
        show(tidy(fig, max(320, 26 * len(pct) + 60)))
        st.caption("Numbers in brackets are cohort sizes. Cohorts under 50 customers are left out.")
    else:
        st.info("Not enough customers in this selection to build cohorts. Widen the filters.")

# ---------- tab 3 ----------
with tab3:
    st.markdown("### Where the revenue comes from")
    seg = segments(df)
    if seg is None:
        st.info("Not enough customers in this selection to group them. Widen the filters.")
    else:
        cols = {"Repeat buyers": HARBOR, "Recent big spenders": "#7FB2C5", "Lapsed big spenders": SIGNAL, "Everyone else": "#B8C4CF"}
        seg = seg.iloc[::-1]
        fig = go.Figure(go.Bar(y=seg.index, x=seg.share, orientation="h", marker_color=[cols[i] for i in seg.index],
                               text=[f"{r.share:.0f}% of revenue, {int(r.customers):,} customers" for r in seg.itertuples()],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:.1f}% of revenue<extra></extra>"))
        fig.update_xaxes(visible=False, range=[0, max(seg.share) * 1.9])
        fig.update_yaxes(showgrid=False)
        show(tidy(fig, 300))
        st.markdown("<p class='note'>Customers are ranked on how recently they ordered and how much they spent (RFM scoring). "
                    "<b>Lapsed big spenders</b> are in the top quarter by spend but have not ordered for a long time. "
                    "They are the first group to win back.</p>", unsafe_allow_html=True)

# ---------- tab 4 ----------
with tab4:
    st.markdown("### The question")
    st.markdown("<p class='note'>Most Olist customers order once and never return. This project asks whether late "
                "deliveries are part of the reason, which customers are worth winning back, and where to start.</p>",
                unsafe_allow_html=True)
    st.markdown("### How it was built")
    st.markdown("- **Data:** the public Olist Brazilian e-commerce dataset on Kaggle, 9 linked tables of real orders.\n"
                "- **SQL:** DuckDB joins orders, customers, items and reviews into one order-level table.\n"
                "- **Analysis:** Pandas for cohorts and RFM scoring, SciPy for the Mann-Whitney U test.\n"
                "- **Dashboard:** Streamlit and Plotly, deployed from the GitHub repository.")
    st.markdown("### What to keep in mind")
    st.markdown("<p class='note'>The data covers one marketplace between 2016 and 2018. The patterns here are associations, "
                "not proof that late delivery causes customers to leave.</p>", unsafe_allow_html=True)
    st.link_button("View the code on GitHub", REPO)