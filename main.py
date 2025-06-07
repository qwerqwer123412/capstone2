# -*- coding: utf-8 -*-
"""
Heterogeneous WNMS Dashboard — Streaming Edition with Prediction vs Ground Truth
- Overview / AP 리스트 / AP 상세 / Station 탭 구성
- AP 리스트 탭: AP 미니 시계열 (Prediction vs Ground Truth)
  • AP 선택 시, 최근 20개 스냅샷을 채워넣고 이후 1초마다 데이터 스트리밍
  • 짝수 인터벌엔 Prediction, 홀수 인터벌엔 Ground Truth 표시
- AP 상세 탭: 선택 AP의 메타 정보 + 최근 20 스냅샷 Prediction vs Ground Truth 시계열 그래프
- Overview 탭: Top10 AP Client 비율, Total AP/Station, AP별 RX/TX/Client Top10 차트
- Station 탭: Station 리스트 + 선택 Station의 최근 20 RSSI 실시간 그래프
"""
from plotly.subplots import make_subplots
import pandas as pd
import torch
import plotly
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 & 전역 메타 정보
# ──────────────────────────────────────────────────────────────────────────────
# 1-1) HeteroData 객체 리스트 로드
PT_PATH = "/home/nsl/WiFi_Network_ML_minjun/WiFi_Network_ML/data/hetero_graph_dataset_weekday_6to16.pt"
dataset = torch.load(PT_PATH, weights_only=False)   # HeteroData 리스트
TS_LIST = [g.timestamp.to_pydatetime() for g in dataset]  # pandas.Timestamp 리스트
N_SNAP = len(dataset)

# 첫 번째 그래프에서 AP/Station 매핑 정보
first = dataset[0]
AP_NAME2IDX = first.ap_name2idx
IDX2AP_NAME = {i: n for n, i in AP_NAME2IDX.items()}
AP_NAMES = [IDX2AP_NAME[i] for i in range(len(AP_NAME2IDX))]
STATION_IP2IDX = first.station_ip2idx
IDX2STATION_IP = {i: ip for ip, i in STATION_IP2IDX.items()}

# 1-2) 트래픽 예측 & 실측 CSV 로드
CSV_PATH = "/mnt/data/lstmcell_results.csv"  # timestamp, AP##_pred, AP##_true, ...
DF = pd.read_csv(CSV_PATH)

if len(TS_LIST) < len(DF):
    raise ValueError(f"TS_LIST 길이({len(TS_LIST)})가 DF 행 수({len(DF)})보다 작습니다.")
elif len(TS_LIST) > len(DF):
    # TS_LIST가 더 길면 DF에 맞춰 자르기
    usable_ts = TS_LIST[:len(DF)]
else:
    usable_ts = TS_LIST

# HeteroData TS_LIST 를 인덱스로 지정
DF.index = pd.to_datetime(usable_ts)
PRED_COL = {
    ap: f"AP{AP_NAME2IDX[ap]}_pred"
    for ap in AP_NAMES
}
TRUE_COL = {
    ap: f"AP{AP_NAME2IDX[ap]}_true"
    for ap in AP_NAMES
}
# Plotly 기본 컬러 팔레트 + 회색보조색
colors = plotly.colors.qualitative.Plotly[:10] + ["#d9d9d9"]
# 카드 공통 스타일
card_style = {
    "backgroundColor": "#ffffff",
    "padding": "16px",
    "borderRadius": "8px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
    "marginBottom": "16px",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dash 앱 초기화 & 레이아웃
# ──────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


app.layout = html.Div([
    dcc.Interval(id="global-interval", interval=90_000, n_intervals=0),
    dcc.Interval(id="overview-interval", interval=5_000, n_intervals=0),
    dcc.Interval(id="stream-interval", interval=1_000, n_intervals=0),
    dcc.Store(id="mini-chart-start-interval", data=None),

    html.H2("Heterogeneous WNMS Dashboard", style={"marginBottom": "20px", "color": "#333"}),
    html.Div(style={"height": "4px", "backgroundColor": "#1890ff", "marginBottom": "20px"}),

    dcc.Tabs(
        id="main-tabs",
        value="overview",
        children=[
            dcc.Tab(label="Overview", value="overview", style={"fontWeight": "bold"}),
            dcc.Tab(label="AP 리스트", value="ap-list", style={"fontWeight": "bold"}),
            dcc.Tab(label="AP 상세", value="ap-detail", style={"fontWeight": "bold"}),
            dcc.Tab(label="Station", value="station", style={"fontWeight": "bold"}),
        ],
        style={"marginBottom": "20px"},
    ),
    html.Div(id="tab-content", style={"marginTop": "10px"}),
], style={"padding": "20px", "backgroundColor": "#f5f7fa", "minHeight": "100vh"})

# ──────────────────────────────────────────────────────────────────────────────
# 3-A. Overview 탭 콘텐츠
# ──────────────────────────────────────────────────────────────────────────────
def render_overview_tab():
    return html.Div(style={"padding": "0 24px"}, children=[
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("Top 10 AP Client 비율", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"}),
                dcc.Graph(id="overview-pie", config={"displayModeBar": False}, style={"height": "300px"}),
            ]),
            html.Div(style={**card_style, "flex": "0 0 200px", "textAlign": "center"}, children=[
                html.Div("Total AP", style={"fontWeight": "bold", "marginBottom": "8px"}),
                html.Div(id="overview-total-ap", style={"fontSize": "36px", "color": "#1890ff"}),
                html.Div(id="overview-ap-status", style={"marginTop": "4px", "color": "#888", "fontSize": "14px"}),
            ]),
            html.Div(style={**card_style, "flex": "0 0 200px", "textAlign": "center"}, children=[
                html.Div("Total Stations", style={"fontWeight": "bold", "marginBottom": "8px"}),
                html.Div(id="overview-total-sta", style={"fontSize": "36px", "color": "#52c41a"}),
                html.Div(id="overview-sta-top10", style={"marginTop": "4px", "color": "#888", "fontSize": "14px"}),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("AP별 Client Count (Top 10)", style={"fontWeight": "bold", "marginBottom": "8px"}),
                dcc.Graph(id="overview-bar-clients", config={"displayModeBar": False}, style={"height": "280px"}),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("AP별 RX Bytes (Top 10)", style={"fontWeight": "bold", "marginBottom": "8px"}),
                dcc.Graph(id="overview-bar-rx", config={"displayModeBar": False}, style={"height": "280px"}),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("AP별 TX Bytes (Top 10)", style={"fontWeight": "bold", "marginBottom": "8px"}),
                dcc.Graph(id="overview-bar-tx", config={"displayModeBar": False}, style={"height": "280px"}),
            ]),
        ]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. AP 리스트 탭 콘텐츠 (Streaming Mini Chart)
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_list_tab():
    return html.Div(style={"padding": "0 24px"}, children=[
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("AP 리스트", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"}),
                dash_table.DataTable(
                    id="ap-table",
                    columns=[
                        {"name": "AP 이름", "id": "name"},
                        {"name": "MAC", "id": "mac"},
                        {"name": "Rx", "id": "rx", "type": "numeric"},
                        {"name": "Tx", "id": "tx", "type": "numeric"},
                        {"name": "클라이언트 수", "id": "clients", "type": "numeric"},
                    ],
                    data=[], page_size=15, row_selectable="single",
                    style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#fafafa", "fontWeight": "bold"},
                    style_cell={"padding": "8px", "textAlign": "center"},
                    style_cell_conditional=[{"if": {"column_id": "name"}, "textAlign": "left"}, {"if": {"column_id": "mac"}, "textAlign": "left"}],
                ),
            ]),
            html.Div(style={**card_style, "flex":"2"}, children=[
                html.Div("Prediction vs Ground Truth (실시간)", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"}),
                dcc.Graph(
                    id="ap-mini-chart",
                    figure={
                        "data": [
                            {"x": [], "y": [], "name": "Prediction",   "mode": "lines+markers"},
                            {"x": [], "y": [], "name": "Ground Truth", "mode": "lines+markers"},
                        ],
                        "layout": {"margin": {"l": 40, "r": 10, "t": 30, "b": 40}, "xaxis": {"title": "Time"}, "yaxis": {"title": "Value"}},
                    },
                    animate=True,
                    style={"height": "320px"},
                ),
            ]),
        ]),
    ])

# ──────────────────────────────────────────────────────────────────────────────
# 3-C. AP 상세 탭 콘텐츠
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_detail_tab():
    return html.Div(style={"padding": "0 24px"}, children=[
        dcc.Dropdown(id="ap-detail-dropdown", options=[{"label": a, "value": a} for a in AP_NAMES], value=AP_NAMES[0], style={"width": "40%", "marginBottom": "16px"}),
        html.Div(id="ap-detail-content"),
    ])

# ──────────────────────────────────────────────────────────────────────────────
# 3-D. Station 탭 콘텐츠
# ──────────────────────────────────────────────────────────────────────────────
def render_station_tab():
    g_last = dataset[-1]
    rows = []
    for ip, idx in STATION_IP2IDX.items():
        if idx < g_last["Station"].x.size(0):
            vec = g_last["Station"].x[idx]
            rows.append({"ip": ip, "rssi": round(vec[0].item(),1), "snr": round(vec[1].item(),1)})
    return html.Div(style={"padding": "0 24px"}, children=[
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.Div("Station 리스트", style={"fontWeight": "bold", "marginBottom": "12px"}),
                dash_table.DataTable(
                    id="station-table",
                    columns=[{"name": "IP", "id": "ip"}, {"name": "RSSI", "id": "rssi"}, {"name": "SNR", "id": "snr"}],
                    data=rows, page_size=15, row_selectable="single",
                    style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#fafafa", "fontWeight": "bold"}, style_cell={"padding": "8px", "textAlign": "center"},
                ),
            ]),
            html.Div(style={**card_style, "flex": "0 0 400px"}, children=[
                html.Div("Station RSSI (최근 20)", style={"fontWeight": "bold", "marginBottom": "12px"}),
                dcc.Graph(id="station-traffic-chart", style={"height": "320px"}),
                html.Div(id="station-roam-history", style={"color": "#888", "marginTop": "16px"}),
            ]),
        ]),
    ])

# ──────────────────────────────────────────────────────────────────────────────
# 4. 탭 전환 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), [Input("main-tabs", "value")])
def tab_router(tab):
    if tab == "overview":
        return render_overview_tab()
    elif tab == "ap-list":
        return render_ap_list_tab()
    elif tab == "ap-detail":
        return render_ap_detail_tab()
    elif tab == "station":
        return render_station_tab()
    else:
        return html.Div("존재하지 않는 탭입니다.", style={"color": "#f5222d"})

# ──────────────────────────────────────────────────────────────────────────────
# 5-1. AP 테이블 갱신 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(Output("ap-table", "data"), [Input("overview-interval", "n_intervals")])
def refresh_ap_table(n_intervals):
    idx = n_intervals % N_SNAP
    g = dataset[idx]
    ap_clients = {ap: 0 for ap in AP_NAMES}
    if ('AP', 'ap_station', 'Station') in g.edge_types:
        ei = g[('AP','ap_station','Station')].edge_index
        for k in range(ei.size(1)):
            ap_clients[IDX2AP_NAME[ei[0,k].item()]] += 1
    rows = []
    for ap in AP_NAMES:
        meta = g['AP'].meta[AP_NAME2IDX[ap]]
        vec = g['AP'].x[AP_NAME2IDX[ap]].cpu().numpy().tolist()
        rows.append({"name":ap, "mac":meta.get('mac',''), "rx":int(vec[1]), "tx":int(vec[2]), "clients":ap_clients[ap]})
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# 5-2. AP 미니 차트 초기화 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "figure"),
    Output("mini-chart-start-interval", "data"),
    Input("ap-table", "selected_rows"),
    State("stream-interval", "n_intervals"),
    State("overview-interval", "n_intervals"),
)
def reset_mini_chart_and_store_start(rows, stream_n_intervals, overview_n_intervals):
    if not rows:
        raise PreventUpdate
    ap_idx = rows[0]
    ap_name = AP_NAMES[ap_idx]
    base_idx = overview_n_intervals % N_SNAP
    start_hist = max(0, base_idx - 19)
    window = list(range(start_hist, base_idx+1))
    times, preds, trues = [], [], []
    for i in window:
        times.append(TS_LIST[i])
        preds.append(DF.iloc[i][PRED_COL[ap_name]])
        trues.append(DF.iloc[i][TRUE_COL[ap_name]])
    fig = go.Figure([
        go.Scatter(x=times, y=preds, name="Prediction", mode="lines+markers"),
        go.Scatter(x=times, y=trues, name="Ground Truth", mode="lines+markers"),
    ]).update_layout(

        title={"text":f"{ap_name} 예측 vs 실측","font":{"size":16,"color":"#333"}},
        margin={"l":40,"r":20,"t":40,"b":40}, xaxis={"title":"Time"},  yaxis=dict(title="Value", range=[5,15]), plot_bgcolor="#fafafa"
    )
    return fig, {"base_idx":base_idx, "stream_start":stream_n_intervals}

# ──────────────────────────────────────────────────────────────────────────────
# 5-3. AP 미니 차트 스트리밍 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "extendData"),
    Input("stream-interval", "n_intervals"),
    State("ap-table", "selected_rows"),
    State("mini-chart-start-interval", "data"),
    prevent_initial_call=True,
)
def stream_mini_point(n_intervals, rows, stored_data):
    if not rows or not stored_data:
        raise PreventUpdate
    ap_name = AP_NAMES[rows[0]]
    base_idx    = stored_data["base_idx"]
    stream_start= stored_data["stream_start"]
    delta       = n_intervals - stream_start
    next_idx    = base_idx + delta//2 + 1
    if next_idx < 0 or next_idx >= N_SNAP:
        raise PreventUpdate
    ts_next = TS_LIST[next_idx]
    pred_val= DF.iloc[next_idx][PRED_COL[ap_name]]
    true_val= DF.iloc[next_idx][TRUE_COL[ap_name]]
    if delta % 2 == 0:
        return ({"x":[[ts_next]],"y":[[pred_val]]}, [0], 20)
    return ({"x":[[ts_next]],"y":[[true_val]]}, [1], 20)

# ──────────────────────────────────────────────────────────────────────────────
# 6-A. Overview 탭 그래프 업데이트 콜백 (변경 없음)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback([
    Output("overview-pie","figure"), Output("overview-total-ap","children"), Output("overview-ap-status","children"),
    Output("overview-total-sta","children"), Output("overview-sta-top10","children"),
    Output("overview-bar-clients","figure"), Output("overview-bar-rx","figure"), Output("overview-bar-tx","figure"),
], [Input("overview-interval","n_intervals")])
def update_overview_graphs(n_intervals):
    current_idx = n_intervals % N_SNAP
    g_last = dataset[current_idx]

    total_ap = len(AP_NAMES)
    up_ap = total_ap
    down_ap = 0

    ap_clients = {ap: 0 for ap in AP_NAMES}
    if ("AP", "ap_station", "Station") in g_last.edge_types:
        ei_as = g_last[("AP", "ap_station", "Station")].edge_index
        for k in range(ei_as.size(1)):
            ap_clients[IDX2AP_NAME[ei_as[0, k].item()]] += 1
    ap_top10_clients = sorted(
        [{"name": k, "clients": v} for k, v in ap_clients.items()],
        key=lambda x: x["clients"],
        reverse=True,
    )[:10]

    rx_idx = 1
    tx_idx = 2
    ap_rx = {ap: g_last["AP"].x[i, rx_idx].item() for i, ap in enumerate(AP_NAMES)}
    ap_tx = {ap: g_last["AP"].x[i, tx_idx].item() for i, ap in enumerate(AP_NAMES)}
    ap_top10_rx = sorted(
        [{"name": k, "rx": v} for k, v in ap_rx.items()], key=lambda x: x["rx"], reverse=True
    )[:10]
    ap_top10_tx = sorted(
        [{"name": k, "tx": v} for k, v in ap_tx.items()], key=lambda x: x["tx"], reverse=True
    )[:10]

    sta_cnt = {ip: 0 for ip in g_last.station_ip2idx.keys()}
    if ("AP", "ap_station", "Station") in g_last.edge_types:
        ei_as = g_last[("AP", "ap_station", "Station")].edge_index
        for k in range(ei_as.size(1)):
            sta_ip = IDX2STATION_IP.get(ei_as[1, k].item())
            if sta_ip in sta_cnt:
                sta_cnt[sta_ip] += 1
    sta_top10 = sorted(
        [{"ip": k, "connections": v} for k, v in sta_cnt.items()],
        key=lambda x: x["connections"],
        reverse=True,
    )[:10]
    total_sta = g_last["Station"].x.size(0)

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=[d["name"] for d in ap_top10_clients] + ["기타"],
                values=[d["clients"] for d in ap_top10_clients]
                + [sum(ap_clients.values()) - sum(d["clients"] for d in ap_top10_clients)],
                hole=0.4,
                marker={"colors": colors},
                textinfo="label+percent",
                insidetextorientation="radial",
            )
        ]
    ).update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20})

    total_ap_str = str(total_ap)
    ap_status_str = f"UP: {up_ap} / DOWN: {down_ap}"
    total_sta_str = str(total_sta)
    sta_top10_str = f"Top10 연결: {sum(d['connections'] for d in sta_top10)}"

    bar_clients_fig = go.Figure(
        data=[
            go.Bar(
                x=[d["name"] for d in ap_top10_clients],
                y=[d["clients"] for d in ap_top10_clients],
                marker={"color": colors},
                text=[d["clients"] for d in ap_top10_clients],
                textposition="auto",
            )
        ]
    ).update_layout(
        xaxis={"title": "AP Name"},
        yaxis={"title": "Client 수"},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )

    bar_rx_fig = go.Figure(
        data=[
            go.Bar(
                x=[d["name"] for d in ap_top10_rx],
                y=[d["rx"] for d in ap_top10_rx],
                marker={"color": colors},
                text=[f"{d['rx']:,}" for d in ap_top10_rx],
                textposition="auto",
            )
        ]
    ).update_layout(
        xaxis={"title": "AP Name"},
        yaxis={"title": "RX (Bytes)"},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )

    bar_tx_fig = go.Figure(
        data=[
            go.Bar(
                x=[d["name"] for d in ap_top10_tx],
                y=[d["tx"] for d in ap_top10_tx],
                marker={"color": colors},
                text=[f"{d['tx']:,}" for d in ap_top10_tx],
                textposition="auto",
            )
        ]
    ).update_layout(
        xaxis={"title": "AP Name"},
        yaxis={"title": "TX (Bytes)"},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )

    return (
        pie_fig,
        total_ap_str,
        ap_status_str,
        total_sta_str,
        sta_top10_str,
        bar_clients_fig,
        bar_rx_fig,
        bar_tx_fig,
    )



@app.callback(
    Output("ap-detail-content", "children"),
    [Input("ap-detail-dropdown", "value"), Input("overview-interval", "n_intervals")]
)
def ap_detail(ap_name, n_intervals):
    if not ap_name:
        return html.Div("AP를 선택하세요.", style={"color":"#888"})

    current_idx = n_intervals % N_SNAP
    start_idx   = max(0, current_idx - 19)
    window      = list(range(start_idx, current_idx+1))

    times, preds, trues = [], [], []
    rxs, txs, clients   = [], [], []

    for i in window:
        g = dataset[i]
        times.append(TS_LIST[i])

        # 예측/실측
        preds.append(DF.iloc[i][PRED_COL[ap_name]])
        trues.append(DF.iloc[i][TRUE_COL[ap_name]])

        # RX/TX
        idx_feat = AP_NAME2IDX[ap_name]
        feat = g['AP'].x[idx_feat]
        rxs.append(feat[1].item())
        txs.append(feat[2].item())

        # Client 수
        cnt = 0
        if ('AP','ap_station','Station') in g.edge_types:
            ei = g[('AP','ap_station','Station')].edge_index
            cnt = (ei[0] == idx_feat).sum().item()
        clients.append(cnt)

    # 서브플롯 3행 생성
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.08,
        subplot_titles=[
            f"{ap_name} 예측 vs 실측",
            "RX & TX Bytes",
            "Client Count"
        ]
    )

    # 1행: Prediction vs True
    fig.add_trace(go.Scatter(x=times, y=preds, name="Prediction", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=trues,  name="Ground Truth", mode="lines+markers"), row=1, col=1)
    fig.update_yaxes(title_text="Value", range=[0,20], row=1, col=1)

    # 2행: RX, TX
    fig.add_trace(go.Scatter(x=times, y=rxs, name="RX Bytes", mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=txs, name="TX Bytes", mode="lines+markers"), row=2, col=1)
    fig.update_yaxes(title_text="Bytes", row=2, col=1)

    # 3행: Clients
    fig.add_trace(go.Scatter(x=times, y=clients, name="Clients", mode="lines+markers"), row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    fig.update_layout(
        height=700,
        margin={"l":40,"r":20,"t":60,"b":40},
        plot_bgcolor="#fafafa"
    )

    # AP 메타 정보 테이블
    g_meta = dataset[current_idx]
    mdx    = AP_NAME2IDX[ap_name]
    meta   = g_meta['AP'].meta[mdx]
    rx_latest = rxs[-1]
    tx_latest = txs[-1]
    client_latest = clients[-1]
    pred_latest = preds[-1]
    true_latest = trues[-1]
    meta_df = pd.DataFrame([
        {'항목': 'AP', '값': ap_name},
        {'항목': 'MAC', '값': meta.get('mac', '')},
        {'항목': 'Latest RX', '값': rx_latest},
        {'항목': 'Latest TX', '값': tx_latest},
        {'항목': 'Latest Clients', '값': client_latest},
        {'항목': 'Latest Prediction', '값': pred_latest},
        {'항목': 'Latest True', '값': true_latest},
    ])
    meta_table = dash_table.DataTable(
        columns=[{"name":"항목","id":"항목"},{"name":"값","id":"값"}],
        data=meta_df.to_dict('records'),
        style_header={'display':'none'},
        style_cell={'padding':'8px','textAlign':'left'}
    )

    return html.Div([
        html.Div(style={**card_style}, children=[
            html.Div("AP 메타 정보", style={"fontWeight":"bold"}),
            meta_table
        ]),
        html.Div(style={**card_style}, children=[
            dcc.Graph(figure=fig)
        ]),
    ])
# ──────────────────────────────────────────────────────────────────────────────
# 7. Station 탭 상세 콜백 (변경 없음)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("station-traffic-chart","figure"), Output("station-roam-history","children"),
    [Input("station-table","selected_rows"), Input("global-interval","n_intervals")]
)
def station_detail(rows, _):
    if not rows or len(rows) == 0:
        return go.Figure(), html.Div("Station을 선택하세요.", style={"color": "#888"})

    ip_list = [ip for ip, idx in STATION_IP2IDX.items() if idx < dataset[-1]["Station"].x.size(0)]
    sel_idx = rows[0]
    if sel_idx >= len(ip_list):
        return go.Figure(), html.Div("올바른 Station이 아닙니다.", style={"color": "#f5222d"})

    selected_ip = ip_list[sel_idx]
    sta_idx     = STATION_IP2IDX[selected_ip]

    window     = range(max(0, N_SNAP - 20), N_SNAP)
    times      = []
    rssi_vals  = []
    for i in window:
        g = dataset[i]
        times.append(TS_LIST[i].to_pydatetime())
        val = 0
        if ('Station', 'station_ap', 'AP') in g.edge_types:
            ei = g[('Station', 'station_ap', 'AP')].edge_index
            ea = g[('Station', 'station_ap', 'AP')].edge_attr
            for k in range(ei.size(1)):
                if ei[0, k].item() == sta_idx:
                    val = ea[k, 0].item()
                    break
        rssi_vals.append(val)

    fig = go.Figure(
        [go.Scatter(x=times, y=rssi_vals, name="RSSI", mode="lines+markers", line={"color": "#fa8c16"})]
    ).update_layout(
        title="최근 20 스냅샷 RSSI",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
        xaxis={"title": "Time"},
        yaxis={"title": "RSSI"},
    )

    return fig, html.Div("로밍 히스토리 데이터가 없습니다.", style={"color": "#888"})


# ──────────────────────────────────────────────────────────────────────────────
# 8. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8051)
