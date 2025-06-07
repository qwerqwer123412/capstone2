# -*- coding: utf-8 -*-
"""
Heterogeneous WNMS Dashboard — Streaming Edition (Full Code with AP 상세 시계열 그래프 추가)
- AP 리스트 탭: AP 미니 시계열(실시간 스트리밍, extendData) + 실시간 테이블 갱신 (Overview와 동일한 인덱스 사용)
  → AP 클릭 시, 최근 20개 스냅샷을 미리 채워놓고 이후 새 데이터가 들어오면 오른쪽으로 슬라이딩
- Overview / AP 상세 / Station 탭: 90초마다 전체 갱신
- AP 상세 탭에 선택된 AP의 metadata 테이블 + 최근 20스냅샷에 대한 Rx/Tx 트래픽, 클라이언트 수 시계열 그래프 추가
"""
import dash
import plotly
import pandas as pd
import torch
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 & 전역 메타 정보
# ──────────────────────────────────────────────────────────────────────────────
PATH = (
    "/home/nsl/WiFi_Network_ML_minjun/WiFi_Network_ML/data/"
    "hetero_graph_dataset_weekday_6to16.pt"
)
dataset        = torch.load(PATH, weights_only=False)   # HeteroData 객체 리스트 불러오기
TS_LIST        = [g.timestamp for g in dataset]         # pandas.Timestamp 리스트
N_SNAP         = len(dataset)

first          = dataset[0]
AP_NAME2IDX    = first.ap_name2idx                                   # {"AP01":0, "AP02":1, ...}
IDX2AP_NAME    = {i: n for n, i in AP_NAME2IDX.items()}             # {0:"AP01", 1:"AP02", ...}
AP_NAMES       = [IDX2AP_NAME[i] for i in range(len(AP_NAME2IDX))]    # ["AP01", "AP02", ...]

STATION_IP2IDX = first.station_ip2idx                                 # {"192.168.0.10":0, ...}
IDX2STATION_IP = {i: ip for ip, i in STATION_IP2IDX.items()}          # {0:"192.168.0.10", ...}

# Plotly 기본 컬러 팔레트 + 회색보조색
colors = plotly.colors.qualitative.Plotly[:10] + ["#d9d9d9"]

# 공통 카드 스타일
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
app    = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        # 1) Overview/AP 상세/Station용 인터벌 (90초 주기)
        dcc.Interval(id="global-interval", interval=90_000, n_intervals=0),
        dcc.Interval(id="overview-interval", interval=5_000, n_intervals=0),
        # 2) AP 미니 차트 실시간 스트리밍용 인터벌 (1초 주기)
        dcc.Interval(id="stream-interval", interval=1_000, n_intervals=0),

        # → Store: {'base_idx': int, 'stream_start': int} 형식으로 저장
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
    ],
    style={"padding": "20px", "backgroundColor": "#f5f7fa", "minHeight": "100vh"},
)

# ──────────────────────────────────────────────────────────────────────────────
# 3-A. Overview 탭 콘텐츠 함수
# ──────────────────────────────────────────────────────────────────────────────
def render_overview_tab():
    return html.Div(
        style={"padding": "0 24px"},
        children=[
            # 상단 카드 그룹
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "Top 10 AP Client 비율",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"},
                            ),
                            dcc.Graph(
                                id="overview-pie",
                                config={"displayModeBar": False},
                                style={"height": "300px"},
                                figure=go.Figure(),
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "0 0 200px", "textAlign": "center"},
                        children=[
                            html.Div(
                                "Total AP",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"},
                            ),
                            html.Div(
                                "-",
                                id="overview-total-ap",
                                style={"fontSize": "36px", "color": "#1890ff"},
                            ),
                            html.Div(
                                "-",
                                id="overview-ap-status",
                                style={"marginTop": "4px", "color": "#888", "fontSize": "14px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "0 0 200px", "textAlign": "center"},
                        children=[
                            html.Div(
                                "Total Stations",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"},
                            ),
                            html.Div(
                                "-",
                                id="overview-total-sta",
                                style={"fontSize": "36px", "color": "#52c41a"},
                            ),
                            html.Div(
                                "-",
                                id="overview-sta-top10",
                                style={"marginTop": "4px", "color": "#888", "fontSize": "14px"},
                            ),
                        ],
                    ),
                ],
            ),
            # 하단 막대 그래프 그룹
            html.Div(
                style={"display": "flex", "gap": "16px"},
                children=[
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "AP별 Client Count (Top 10)",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"},
                            ),
                            dcc.Graph(
                                id="overview-bar-clients",
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                                figure=go.Figure(),
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "AP별 RX Bytes (Top 10)",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"},
                            ),
                            dcc.Graph(
                                id="overview-bar-rx",
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                                figure=go.Figure(),
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "AP별 TX Bytes (Top 10)",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "8px"},
                            ),
                            dcc.Graph(
                                id="overview-bar-tx",
                                config={"displayModeBar": False},
                                style={"height": "280px"},
                                figure=go.Figure(),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. AP 리스트 탭 콘텐츠 (Streaming Mini Chart + 실시간 테이블 갱신)
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_list_tab():
    return html.Div(
        style={"padding": "0 24px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "16px"},
                children=[
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "AP 리스트",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"},
                            ),
                            dash_table.DataTable(
                                id="ap-table",
                                columns=[
                                    {"name": "AP 이름",       "id": "name"},
                                    {"name": "MAC",          "id": "mac"},
                                    {"name": "Rx",           "id": "rx",      "type": "numeric"},
                                    {"name": "Tx",           "id": "tx",      "type": "numeric"},
                                    {"name": "클라이언트 수", "id": "clients", "type": "numeric"},
                                ],
                                data=[],
                                page_size=15,
                                row_selectable="single",
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#fafafa",
                                    "fontWeight": "bold",
                                    "borderBottom": "1px solid #ddd",
                                },
                                style_cell={
                                    "padding": "8px",
                                    "textAlign": "center",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "name"}, "textAlign": "left"},
                                    {"if": {"column_id": "mac"},  "textAlign": "left"},
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "0 0 400px"},
                        children=[
                            html.Div(
                                "AP 미니 시계열 (실시간)",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"},
                            ),
                            dcc.Graph(
                                id="ap-mini-chart",
                                figure={
                                    "data": [
                                        {"x": [], "y": [], "name": "Rx", "mode": "lines+markers", "line": {"color": "#1890ff"}},
                                        {"x": [], "y": [], "name": "Tx", "mode": "lines+markers", "line": {"color": "#52c41a"}},
                                    ],
                                    "layout": {
                                        "margin": {"l": 40, "r": 10, "t": 30, "b": 40},
                                        "xaxis": {"title": "Time", "autorange": True, "tickfont": {"size": 11}},
                                        "yaxis": {"title": "Bytes", "tickfont": {"size": 11}},
                                    },
                                },
                                animate=True,
                                style={"height": "320px"},
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-C. AP 상세 탭 콘텐츠 (기본값: 첫 번째 AP 선택 + 시계열 그래프 포함)
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_detail_tab():
    return html.Div(
        style={"padding": "0 24px"},
        children=[
            html.Div(
                dcc.Dropdown(
                    id="ap-detail-dropdown",
                    options=[{"label": a, "value": a} for a in AP_NAMES],
                    value=AP_NAMES[0],
                    style={"width": "40%", "marginBottom": 16},
                )
            ),
            html.Div(id="ap-detail-content", style={"marginTop": "16px"}),
        ]
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3-D. Station 탭 콘텐츠
# ──────────────────────────────────────────────────────────────────────────────
def render_station_tab():
    g_last = dataset[-1]

    rows = []
    for ip, idx in STATION_IP2IDX.items():
        if idx < g_last["Station"].x.size(0):
            vec          = g_last["Station"].x[idx]
            rssi, snr, spd, lbl = (vec[i].item() for i in range(4))
            connected_ap = ""
            if ('Station', 'station_ap', 'AP') in g_last.edge_types:
                ei_sa = g_last[('Station', 'station_ap', 'AP')].edge_index
                for k in range(ei_sa.size(1)):
                    if ei_sa[0, k].item() == idx:
                        connected_ap = IDX2AP_NAME[ei_sa[1, k].item()]
                        break
            rows.append({
                "ip": ip,
                "connected_ap": connected_ap,
                "rssi": round(rssi, 1),
                "snr": round(snr, 1),
                "speed": round(spd, 1),
                "label": int(lbl),
            })

    return html.Div(
        style={"padding": "0 24px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "16px"},
                children=[
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            html.Div(
                                "Station 리스트",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"},
                            ),
                            dash_table.DataTable(
                                id="station-table",
                                columns=[
                                    {"name": "Station IP", "id": "ip"},
                                    {"name": "연결 AP",     "id": "connected_ap"},
                                    {"name": "RSSI",       "id": "rssi"},
                                    {"name": "SNR",        "id": "snr"},
                                    {"name": "Speed",      "id": "speed"},
                                    {"name": "Label",      "id": "label"},
                                ],
                                data=rows,
                                page_size=15,
                                row_selectable="single",
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#fafafa",
                                    "fontWeight": "bold",
                                    "borderBottom": "1px solid #ddd",
                                },
                                style_cell={
                                    "padding": "8px",
                                    "textAlign": "center",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "ip"},           "textAlign": "left"},
                                    {"if": {"column_id": "connected_ap"}, "textAlign": "left"},
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "0 0 400px"},
                        children=[
                            html.Div(
                                "Station RSSI (최근 20 스냅샷)",
                                style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"},
                            ),
                            dcc.Graph(id="station-traffic-chart", style={"height": "320px"}),
                            html.Div(id="station-roam-history", style={"marginTop": "16px", "color": "#888"}),
                        ],
                    ),
                ],
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. 탭 전환 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    [ Input("main-tabs", "value") ]
)
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
# AP 테이블 데이터 실시간(Overview와 동일 인덱스) 갱신 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-table", "data"),
    [ Input("overview-interval", "n_intervals") ]
)
def refresh_ap_table(n_intervals):
    current_idx = n_intervals % N_SNAP
    g_last = dataset[current_idx]

    ap_clients = {ap: 0 for ap in AP_NAMES}
    if ('AP', 'ap_station', 'Station') in g_last.edge_types:
        ei_as = g_last[('AP', 'ap_station', 'Station')].edge_index
        for k in range(ei_as.size(1)):
            ap_clients[IDX2AP_NAME[ei_as[0, k].item()]] += 1

    rows = []
    for ap in AP_NAMES:
        idx = AP_NAME2IDX[ap]
        vec = g_last["AP"].x[idx].cpu().numpy().tolist()
        rows.append({
            "name": ap,
            "mac": g_last["AP"].meta[idx].get("mac", ""),
            "rx": int(vec[1]),
            "tx": int(vec[2]),
            "clients": ap_clients[ap],
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 5-1. AP 클릭 시: 미니 차트 초기화 (최근 20개 채워넣기) + 시작 인터벌 기록
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "figure"),
    Output("mini-chart-start-interval", "data"),
    Input("ap-table", "selected_rows"),
    State("stream-interval", "n_intervals"),
    State("overview-interval", "n_intervals"),
)
def reset_mini_chart_and_store_start(rows, stream_n_intervals, overview_n_intervals):
    if not rows or len(rows) == 0:
        raise PreventUpdate

    ap_row_idx = rows[0]
    base_idx = overview_n_intervals % N_SNAP
    start_hist = max(0, base_idx - 19)
    window = list(range(start_hist, base_idx + 1))

    times = []
    rx_vals = []
    tx_vals = []
    for i in window:
        times.append(TS_LIST[i].to_pydatetime())
        g = dataset[i]
        rx_vals.append(g["AP"].x[ap_row_idx][1].item())
        tx_vals.append(g["AP"].x[ap_row_idx][2].item())

    fig = go.Figure(
        data=[
            go.Scatter(x=times, y=rx_vals, name="Rx", mode="lines+markers", line={"color": "#1890ff"}),
            go.Scatter(x=times, y=tx_vals, name="Tx", mode="lines+markers", line={"color": "#52c41a"}),
        ],
        layout=dict(
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Time", "autorange": True, "tickfont": {"size": 11}},
            yaxis={"title": "Bytes", "tickfont": {"size": 11}},
        ),
    )

    data_to_store = {
        "base_idx": base_idx,
        "stream_start": stream_n_intervals if stream_n_intervals is not None else 0
    }
    return fig, data_to_store


# ──────────────────────────────────────────────────────────────────────────────
# 5-2. AP 미니 차트 스트리밍 (슬라이딩 윈도우: maxPoints=20)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "extendData"),
    Input("stream-interval", "n_intervals"),
    State("ap-table", "selected_rows"),
    State("mini-chart-start-interval", "data"),
    prevent_initial_call=True,
)
def stream_mini_point(n_intervals, rows, stored_data):
    if not rows or len(rows) == 0 or not stored_data:
        raise PreventUpdate

    base_idx = stored_data["base_idx"]
    stream_start = stored_data["stream_start"]
    delta = n_intervals - stream_start
    next_idx = base_idx + delta + 1

    if next_idx < 0 or next_idx >= N_SNAP:
        raise PreventUpdate

    g_next = dataset[next_idx]
    ts_next = TS_LIST[next_idx].to_pydatetime()
    ap_row_idx = rows[0]

    rx_val = g_next["AP"].x[ap_row_idx][1].item()
    tx_val = g_next["AP"].x[ap_row_idx][2].item()

    return (
        {"x": [[ts_next], [ts_next]], "y": [[rx_val], [tx_val]]},
        [0, 1],
        20
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5-3. Overview 탭 그래프 업데이트 콜백
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    [
        Output("overview-pie", "figure"),
        Output("overview-total-ap", "children"),
        Output("overview-ap-status", "children"),
        Output("overview-total-sta", "children"),
        Output("overview-sta-top10", "children"),
        Output("overview-bar-clients", "figure"),
        Output("overview-bar-rx", "figure"),
        Output("overview-bar-tx", "figure"),
    ],
    [ Input("overview-interval", "n_intervals") ],
)
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

    # Pie 차트
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
                showlegend=False,
            )
        ]
    ).update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        annotations=[{"text": "Clients", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 14}}],
    )

    total_ap_str = str(total_ap)
    ap_status_str = f"UP: {up_ap} / DOWN: {down_ap}"
    total_sta_str = str(total_sta)
    sta_top10_str = f"Top10 연결: {sum(d['connections'] for d in sta_top10)}"

    # Bar 차트 (Clients)
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
        xaxis={"title": "", "tickfont": {"size": 11}},
        yaxis={"title": "Client 수", "tickfont": {"size": 11}},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        plot_bgcolor="#fafafa",
    )

    # Bar 차트 (RX)
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
        xaxis={"title": "", "tickfont": {"size": 11}},
        yaxis={"title": "RX (Bytes)", "tickfont": {"size": 11}},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        plot_bgcolor="#fafafa",
    )

    # Bar 차트 (TX)
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
        xaxis={"title": "", "tickfont": {"size": 11}},
        yaxis={"title": "TX (Bytes)", "tickfont": {"size": 11}},
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        plot_bgcolor="#fafafa",
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


# ──────────────────────────────────────────────────────────────────────────────
# 6-A. AP 상세 탭 콜백 (90초마다 갱신)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-detail-content", "children"),
    [Input("ap-detail-dropdown", "value"), Input("overview-interval", "n_intervals")]
)
def ap_detail(ap_name, n_intervals):
    if not ap_name:
        return html.Div("AP를 선택하세요.", style={"color": "#888"})

    current_idx = n_intervals % N_SNAP
    g = dataset[current_idx]

    idx = AP_NAME2IDX[ap_name]
    vec = g["AP"].x[idx].cpu().numpy().tolist()
    rx0, tx0, chan, x_c, y_c = vec[0], vec[1], int(vec[3]), vec[-2], vec[-1]

    # 메타 정보 DataFrame으로 정리
    meta_df = pd.DataFrame.from_dict(
        {
            "항목": ["AP", "MAC", "채널", "x_coord", "y_coord", "최근 Rx", "최근 Tx"],
            "값":   [ap_name, g["AP"].meta[idx].get("mac", ""), chan, f"{x_c:.2f}", f"{y_c:.2f}", int(rx0), int(tx0)],
        }
    )

    # DataTable로 메타 출력
    meta_table = dash_table.DataTable(
        columns=[{"name": "항목", "id": "항목"}, {"name": "값", "id": "값"}],
        data=meta_df.to_dict("records"),
        style_header={"display": "none"},
        style_cell={
            "padding": "8px 12px",
            "fontSize": "14px",
            "border": "none",
            "textAlign": "left",
        },
        style_cell_conditional=[{"if": {"column_id": "값"}, "textAlign": "right"}],
    )

    # 최근 20 스냅샷 데이터 (current_idx 기준으로 뒤에서 20개)
    start_idx = max(0, current_idx - 19)
    window = range(start_idx, current_idx + 1)

    times = []
    rx_vals = []
    tx_vals = []
    client_counts = []

    for i in window:
        g_i = dataset[i]
        times.append(TS_LIST[i].to_pydatetime())
        rx_vals.append(g_i["AP"].x[idx][1].item())
        tx_vals.append(g_i["AP"].x[idx][2].item())

        cnt = 0
        if ("AP", "ap_station", "Station") in g_i.edge_types:
            ei_as = g_i[("AP", "ap_station", "Station")].edge_index
            for k in range(ei_as.size(1)):
                if ei_as[0, k].item() == idx:
                    cnt += 1
        client_counts.append(cnt)

    # Rx/Tx 트래픽 선 그래프
    traffic_fig = go.Figure(
        data=[
            go.Scatter(
                x=times, y=rx_vals, name="Rx", mode="lines+markers", line={"color": "#1890ff"}
            ),
            go.Scatter(
                x=times, y=tx_vals, name="Tx", mode="lines+markers", line={"color": "#52c41a"}
            ),
        ]
    ).update_layout(
        title={
            "text": "최근 20 스냅샷 Rx/Tx 트래픽",
            "font": {"size": 16, "color": "#333"}
        },
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        xaxis={"title": "Time", "tickfont": {"size": 11}},
        yaxis={"title": "Bytes", "tickfont": {"size": 11}},
        plot_bgcolor="#fafafa",
    )

    # Client 수 선 그래프
    client_fig = go.Figure(
        data=[
            go.Scatter(
                x=times, y=client_counts, name="Client Count", mode="lines+markers", line={"color": "#fa8c16"}
            )
        ]
    ).update_layout(
        title={
            "text": "최근 20 스냅샷 클라이언트 수",
            "font": {"size": 16, "color": "#333"}
        },
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        xaxis={"title": "Time", "tickfont": {"size": 11}},
        yaxis={"title": "Clients", "tickfont": {"size": 11}},
        plot_bgcolor="#fafafa",
    )

    return html.Div(
        children=[
            html.Div(
                style={**card_style},
                children=[
                    html.Div("AP 메타 정보", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px"}),
                    meta_table,
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            dcc.Graph(figure=traffic_fig, style={"height": "300px"})
                        ],
                    ),
                    html.Div(
                        style={**card_style, "flex": "1"},
                        children=[
                            dcc.Graph(figure=client_fig, style={"height": "300px"})
                        ],
                    ),
                ],
            ),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6-B. Station 탭 콜백 (90초마다 갱신)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("station-traffic-chart", "figure"),
    Output("station-roam-history", "children"),
    Input("station-table", "selected_rows"),
    Input("global-interval", "n_intervals"),
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
        title={
            "text": "최근 20 스냅샷 RSSI",
            "font": {"size": 16, "color": "#333"}
        },
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        xaxis={"title": "Time", "tickfont": {"size": 11}},
        yaxis={"title": "RSSI", "tickfont": {"size": 11}},
        plot_bgcolor="#fafafa",
    )

    return fig, html.Div("로밍 히스토리 데이터가 없습니다.", style={"color": "#888"})


# ──────────────────────────────────────────────────────────────────────────────
# 7. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8051)
