# src/dashboard.py
import os
import io
import base64
from datetime import timedelta

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import nltk

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from src.utils import load_sentiment_csv

# Ensure stopwords available (safe if already present)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    import nltk
    nltk.download("stopwords")

CSV_PATH = "data/tweets_with_sentiment.csv"

# ---------- Helpers ----------
SENTIMENT_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
SENTIMENT_COLORS = {
    "NEGATIVE": "#EF553B",
    "NEUTRAL":  "#636EFA",
    "POSITIVE": "#00CC96",
}

def compute_rsi(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    pos = (df["sentiment"] == "POSITIVE").sum()
    neg = (df["sentiment"] == "NEGATIVE").sum()
    tot = len(df)
    return (pos - neg) / tot if tot else 0.0

def aggregate_time(df: pd.DataFrame, freq: str = "T") -> pd.DataFrame:
    if df.empty:
        idx = pd.date_range(pd.Timestamp.utcnow().floor(freq), periods=1, freq=freq)
        return pd.DataFrame(index=idx)
    g = (
        df.set_index("timestamp")
          .groupby([pd.Grouper(freq=freq), "sentiment"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=SENTIMENT_ORDER, fill_value=0)
          .sort_index()
    )
    g["TOTAL"] = g.sum(axis=1)
    # Rolling RSI (centered not ideal for streaming; use simple trailing window)
    window = max(3, min(30, int(len(g) * 0.2)))  # adaptive
    pos = g.get("POSITIVE", pd.Series(0, index=g.index))
    neg = g.get("NEGATIVE", pd.Series(0, index=g.index))
    rsi = (pos - neg) / g["TOTAL"].replace({0: np.nan})
    g["RSI"] = rsi.rolling(window=window, min_periods=1).mean().fillna(0)
    return g

def wordcloud_image(texts: pd.Series) -> str:
    text_blob = " ".join(texts.dropna().astype(str))
    if not text_blob.strip():
        text_blob = "no data"
    wc = WordCloud(width=900, height=450, background_color="white",
                   stopwords=STOPWORDS, collocations=False).generate(text_blob)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def top_tokens(df: pd.DataFrame, col: str, k: int = 15):
    # col is "hashtags" or "mentions" (lists)
    if df.empty: return pd.DataFrame(columns=["token","count"])
    s = df[col].explode().dropna()
    s = s[s.str.len() > 1]
    return s.value_counts().head(k).reset_index().rename(columns={"index":"token", col:"count"})

def kpi_card(title: str, value: str, sub: str = "", color: str = "primary"):
    return dbc.Card(
        dbc.CardBody([
            html.I(className=f"fas fa-chart-line text-{color} mb-2", style={"fontSize": "1.5rem"}),
            html.div(title, className="text-sm text-muted fw-bold"),
            html.h2(value, className="text-2xl fw-bold mb-0 text-dark"),
            html.div(sub, className="text-xs text-muted mt-1"),
        ]),
        className="shadow-sm rounded-3 border-0 h-100",
        style={"transition": "transform 0.2s, box-shadow 0.2s", "cursor": "pointer"},
        id={"type": "kpi-card", "index": title}
    )

# ---------- App ----------
app: Dash = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="Social Media Sentiment Analyzer"
)

# Custom CSS for hover effects
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.12) !important;
                transition: all 0.2s ease;
            }
            .form-control:focus, .form-select:focus {
                border-color: #0d6efd;
                box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
            }
            .badge {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
app.layout = dbc.Container([
    dcc.Store(id="store-hashtags"),
    dcc.Interval(id="interval-refresh", interval=10_000, n_intervals=0),  # 10s
    dcc.Loading(id="loading-main", type="default", children=html.Div(id="loading-output")),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fas fa-chart-line me-3 text-primary", style={"fontSize": "2rem"}),
                html.Div([
                    html.H2("📊 Social Media Sentiment Analyzer", className="mt-3 mb-1 text-primary"),
                    html.Div("🚀 Live rolling analytics for campaign hashtags", className="text-muted mb-3"),
                ], className="d-inline-block")
            ], className="d-flex align-items-center")
        ], md=10),
        dbc.Col([
            html.Div([
                dbc.Badge(id="status-badge", color="success", className="mt-4 fs-6", pill=True),
                html.Div(id="last-updated", className="text-xs text-muted mt-1")
            ])
        ], md=2, className="text-end"),
    ]),

    # Controls Panel
    dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-filter me-2"),
            "Filters & Controls"
        ], className="bg-light fw-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("🏷️ Hashtag Filter", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.InputGroupText("#"),
                        dbc.Input(id="input-hashtag", placeholder="YourCampaignHashtag", debounce=True)
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Label("🔍 Text Search", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.InputGroupText(html.I(className="fas fa-search")),
                        dbc.Input(id="input-search", placeholder="launch, price, bug...", debounce=True)
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Label("📊 Confidence Threshold", className="fw-bold"),
                    html.Div(id="conf-value", className="text-primary fw-bold mb-1"),
                    dcc.Slider(id="slider-conf", min=0.0, max=1.0, step=0.05, value=0.50,
                               marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                               tooltip={"always_visible": True, "placement": "bottom"})
                ], md=3),
                dbc.Col([
                    dbc.Label("⏱️ Time Granularity", className="fw-bold"),
                    dbc.RadioItems(
                        id="radio-gran",
                        options=[
                            {"label": "📍 Minute", "value": "T"},
                            {"label": "🕐 Hour", "value": "H"},
                            {"label": "📅 Day", "value": "D"}
                        ],
                        value="T",
                        inline=True,
                        className="mt-2"
                    ),
                ], md=3),
            ], className="g-3"),
        ])
    ], className="mb-3 shadow-sm"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-calendar-alt me-2 text-info"),
                        dbc.Label("📅 Date Range", className="fw-bold mb-2")
                    ]),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=pd.Timestamp("2023-01-01"),
                        max_date_allowed=pd.Timestamp.utcnow().date(),
                        start_date=(pd.Timestamp.utcnow() - pd.Timedelta(days=2)).date(),
                        end_date=pd.Timestamp.utcnow().date(),
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"}
                    )
                ])
            ], className="h-100 shadow-sm")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-sync-alt me-2 text-success"),
                        dbc.Label("🔄 Auto-refresh", className="fw-bold mb-2")
                    ]),
                    html.Div(id="refresh-value", className="text-success fw-bold mb-1"),
                    dcc.Slider(id="slider-refresh", min=5, max=60, step=5, value=10,
                               marks={i: f"{i}s" for i in range(5, 61, 15)})
                ])
            ], className="h-100 shadow-sm")
        ], md=6),
    ], className="g-3 mb-3"),

    # KPIs
    dcc.Loading(
        dbc.Row(id="row-kpis", className="g-3 mt-1"),
        type="dot"
    ),

    # Charts with enhanced styling
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Sentiment Timeline", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="ts-stacked"), type="graph"))
            ], className="shadow-sm h-100")
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🍰 Distribution", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="pie-dist"), type="circle"))
            ], className="shadow-sm h-100")
        ], md=4),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🕐 Hourly Activity", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="bar-hourly"), type="circle"))
            ], className="shadow-sm h-100")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("☁️ Word Cloud", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(
                    html.Img(id="img-wordcloud", style={"width":"100%","height":"auto","borderRadius":"8px"}),
                    type="dot"
                ))
            ], className="shadow-sm h-100")
        ], md=6),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🏷️ Top Hashtags", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="bar-hashtags"), type="circle"))
            ], className="shadow-sm h-100")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💬 Top Mentions", className="fw-bold bg-light"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="bar-mentions"), type="circle"))
            ], className="shadow-sm h-100")
        ], md=6),
    ], className="g-3 mb-3"),

    # Table
    dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-table me-2"),
            "📱 Recent Tweets",
            dbc.Badge(id="tweet-count-badge", className="ms-2")
        ], className="bg-light fw-bold"),
        dbc.CardBody([
    dcc.Loading(
        dash_table.DataTable(
        id="table-tweets",
        columns=[
            {"name":"Time (UTC)","id":"timestamp","type":"datetime"},
            {"name":"Sentiment","id":"sentiment","type":"text"},
            {"name":"Confidence","id":"confidence","type":"numeric","format":dash_table.FormatTemplate.percentage(2)},
            {"name":"Text","id":"text","type":"text"},
        ],
        page_size=10,
        style_cell={"textAlign":"left","whiteSpace":"normal","height":"auto"},
        style_header={"fontWeight":"bold"},
        style_data_conditional=[
            {
                "if": {"filter_query": "{sentiment} = 'POSITIVE'"},
                "backgroundColor": "rgba(0,204,150,0.08)"
            },
            {
                "if": {"filter_query": "{sentiment} = 'NEGATIVE'"},
                "backgroundColor": "rgba(239,85,59,0.08)"
            },
            {
                "if": {"filter_query": "{sentiment} = 'NEUTRAL'"},
                "backgroundColor": "rgba(99,110,250,0.08)"
            },
        ],
            ),
            type="default"
        )]),
    ], className="shadow-sm"),

    html.Div(className="mb-4")
], fluid=True)

# ---------- Callbacks ----------
@app.callback(
    Output("interval-refresh", "interval"),
    Output("refresh-value", "children"),
    Input("slider-refresh", "value")
)
def set_refresh_interval(seconds):
    return int(seconds) * 1000, f"Every {seconds} seconds"

@app.callback(
    Output("conf-value", "children"),
    Input("slider-conf", "value")
)
def update_conf_display(value):
    return f"≥ {value:.0%} confidence"

@app.callback(
    Output("tweet-count-badge", "children"),
    Output("tweet-count-badge", "color"),
    Input("table-tweets", "data")
)
def update_tweet_badge(data):
    count = len(data) if data else 0
    color = "success" if count > 50 else "warning" if count > 10 else "secondary"
    return f"{count} tweets", color

def apply_filters(df, hashtag, search, conf, start_date, end_date):
    if df.empty: return df
    # Date range filter
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date).tz_localize("UTC")]
    if end_date:
        df = df[df["timestamp"] <= (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")]
    # Confidence
    df = df[df["confidence"].fillna(0) >= float(conf)]
    # Hashtag exact (case-insensitive)
    if hashtag and hashtag.strip():
        tag = hashtag.strip().lower()
        df = df[df["text"].str.lower().str.contains(rf"(?<!\w){pd.re.escape(tag)}(?!\w)", regex=True, na=False)]
    # Text search contains
    if search and search.strip():
        s = search.strip().lower()
        df = df[df["text"].str.lower().str.contains(pd.re.escape(s), na=False)]
    return df

@app.callback(
    Output("loading-output", "children"),
    Output("status-badge", "children"),
    Output("status-badge", "color"),
    Output("last-updated", "children"),
    Output("row-kpis", "children"),
    Output("ts-stacked", "figure"),
    Output("pie-dist", "figure"),
    Output("bar-hourly", "figure"),
    Output("img-wordcloud", "src"),
    Output("bar-hashtags", "figure"),
    Output("bar-mentions", "figure"),
    Output("table-tweets", "data"),
    Input("interval-refresh", "n_intervals"),
    Input("input-hashtag", "value"),
    Input("input-search", "value"),
    Input("slider-conf", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("radio-gran", "value"),
)
def update_dashboard(_, hashtag, search, conf, start_date, end_date, gran):
    df = load_sentiment_csv(CSV_PATH)

    # Apply filters
    fdf = apply_filters(df, hashtag, search, conf, start_date, end_date)

    # KPIs
    total = len(fdf)
    pos = (fdf["sentiment"] == "POSITIVE").sum()
    neu = (fdf["sentiment"] == "NEUTRAL").sum()
    neg = (fdf["sentiment"] == "NEGATIVE").sum()
    avg_conf = fdf["confidence"].mean() if total else 0.0
    rsi_now = compute_rsi(fdf)

    kpis = dbc.Row([
        dbc.Col(kpi_card("Total Tweets", f"{total:,}", f"Analyzed tweets", "info"), md=3),
        dbc.Col(kpi_card("Positive", f"{(pos/total*100 if total else 0):.1f}%", f"{pos} tweets", "success"), md=3),
        dbc.Col(kpi_card("Neutral", f"{(neu/total*100 if total else 0):.1f}%", f"{neu} tweets", "secondary"), md=3),
        dbc.Col(kpi_card("Negative", f"{(neg/total*100 if total else 0):.1f}%", f"{neg} tweets", "danger"), md=3),
    ], className="g-3")

    # Time series
    agg = aggregate_time(fdf, freq=gran)
    ts_traces = []
    for snt in SENTIMENT_ORDER:
        if snt in agg.columns:
            ts_traces.append(go.Bar(
                x=agg.index, y=agg[snt], name=snt, marker_color=SENTIMENT_COLORS[snt], opacity=0.9
            ))
    # RSI overlay
    ts_traces.append(go.Scatter(
        x=agg.index, y=agg["RSI"], name="Rolling Sentiment Index",
        mode="lines", line={"width":2, "dash":"solid"}, yaxis="y2"
    ))
    ts_layout = go.Layout(
        barmode="stack",
        title="Sentiment Over Time",
        xaxis={"title":"Time"},
        yaxis={"title":"Count"},
        yaxis2={"title":"RSI", "overlaying":"y", "side":"right", "range":[-1,1]},
        legend={"orientation":"h"},
        margin={"t":40, "l":50, "r":50, "b":40},
    )
    fig_ts = go.Figure(data=ts_traces, layout=ts_layout)

    # Pie distribution
    pie_vals = [neg, neu, pos]
    pie_fig = go.Figure(data=[go.Pie(
        labels=SENTIMENT_ORDER, values=[neg, neu, pos],
        marker={"colors":[SENTIMENT_COLORS[c] for c in SENTIMENT_ORDER]},
        hole=0.35
    )])
    pie_fig.update_layout(title="Sentiment Distribution", margin={"t":40, "l":20, "r":20, "b":20})

    # Hourly bar (based on filtered df)
    if not fdf.empty:
        hour_counts = (fdf.set_index("timestamp")
                         .groupby(fdf["timestamp"].dt.floor("H"))
                         .size())
        fig_hour = go.Figure(data=[go.Bar(x=hour_counts.index, y=hour_counts.values)])
        fig_hour.update_layout(title="Tweets per Hour", xaxis_title="Hour (UTC)", yaxis_title="Tweets")
    else:
        fig_hour = go.Figure()
        fig_hour.update_layout(title="Tweets per Hour")

    # Word cloud
    wc_src = wordcloud_image(fdf["text"]) if not fdf.empty else wordcloud_image(pd.Series([""]))

    # Top hashtags / mentions
    def _bar_for(df_counts, title):
        if df_counts.empty:
            fig = go.Figure(); fig.update_layout(title=title)
            return fig
        fig = go.Figure(data=[go.Bar(x=df_counts["token"], y=df_counts["count"])])
        fig.update_layout(title=title, xaxis_tickangle=-30, margin={"b":80})
        return fig

    bar_hashtags = _bar_for(
        (fdf.assign(x=1)
            .explode("hashtags")
            .dropna(subset=["hashtags"])
            .groupby("hashtags")["x"].count()
            .sort_values(ascending=False).head(15)
            .reset_index().rename(columns={"hashtags":"token","x":"count"})),
        "Top Hashtags"
    )

    bar_mentions = _bar_for(
        (fdf.assign(x=1)
            .explode("mentions")
            .dropna(subset=["mentions"])
            .groupby("mentions")["x"].count()
            .sort_values(ascending=False).head(15)
            .reset_index().rename(columns={"mentions":"token","x":"count"})),
        "Top Mentions"
    )

    # Table data (most recent first)
    table_df = fdf.sort_values("timestamp", ascending=False)\
                  .head(100)[["timestamp","sentiment","confidence","text"]]
    # Confidence 0-1 to 0-100%
    table_df["confidence"] = (table_df["confidence"].fillna(0) * 100).round(1)

    status_text = f"✅ Live • {total:,} tweets"
    status_color = "success" if total > 0 else "warning"
    last_updated = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
    
    return "", status_text, status_color, last_updated, kpis, fig_ts, pie_fig, fig_hour, wc_src, bar_hashtags, bar_mentions, table_df.to_dict("records")

if __name__ == "__main__":
    # Create folders if missing
    os.makedirs("data", exist_ok=True)
    app.run_server(host="0.0.0.0", port=8050, debug=True)

