# -*- coding: utf-8 -*-
"""
Heterogeneous WNMS Dashboard — Streaming Edition (Full Code with AP 상세 시계열 그래프 추가)
- AP 리스트 탭: AP 미니 시계열(실시간 스트리밍, extendData)
- Overview / AP 상세 / Station 탭: 90초마다 전체 갱신
- AP 상세 탭에 선택된 AP의 metadata 테이블 + 최근 20스냅샷에 대한 Rx/Tx 트래픽, 클라이언트 수 시계열 그래프 추가
"""
# 필요
# pandas, PyG
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
    "./data/"
    "hetero_graph_dataset_weekday_6to16.pt"
)
dataset        = torch.load(PATH, weights_only=False)  # HeteroData 객체 리스트 불러오기
TS_LIST        = [g.timestamp for g in dataset]       # pandas.Timestamp 리스트
N_SNAP         = len(dataset)

first          = dataset[0]
AP_NAME2IDX    = first.ap_name2idx                                 # {"AP01":0, "AP02":1, ...}
IDX2AP_NAME    = {i: n for n, i in AP_NAME2IDX.items()}           # {0:"AP01", 1:"AP02", ...}
AP_NAMES       = [IDX2AP_NAME[i] for i in range(len(AP_NAME2IDX))]  # ["AP01", "AP02", ...]

STATION_IP2IDX = first.station_ip2idx                               # {"192.168.0.10":0, ...}
IDX2STATION_IP = {i: ip for ip, i in STATION_IP2IDX.items()}        # {0:"192.168.0.10", ...}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dash 앱 초기화 & 레이아웃
# ──────────────────────────────────────────────────────────────────────────────
app    = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        # 1) Overview/AP 상세/Station용 인터벌 (90초 주기)
        dcc.Interval(id="global-interval", interval=1_000, n_intervals=0),
        # 2) AP 미니 차트 스트리밍용 인터벌 (1초 주기)
        dcc.Interval(id="stream-interval", interval=1_000, n_intervals=0),

        # → Store: AP 선택 시점의 n_intervals 값을 저장
        dcc.Store(id="mini-chart-start-interval", data=None),

        html.H2("Heterogeneous WNMS Dashboard", style={"marginBottom": "20px"}),

        dcc.Tabs(
            id="main-tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview"),
                dcc.Tab(label="AP 리스트", value="ap-list"),
                dcc.Tab(label="AP 상세", value="ap-detail"),
                dcc.Tab(label="Station", value="station"),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "20px"}),
    ],
    style={"padding": "20px"},
)

# ──────────────────────────────────────────────────────────────────────────────
# 3-A. Overview 탭 콘텐츠 함수
# ──────────────────────────────────────────────────────────────────────────────
def render_overview_tab():
    g_last   = dataset[-1]
    total_ap = len(AP_NAMES)
    up_ap    = total_ap
    down_ap  = 0  # 실제 UP/DOWN 데이터가 없으므로 모두 UP으로 가정

    # ──────────────────────────────────────────────────────────────────────────
    # ① AP별 클라이언트 수 집계 (기존)
    # ──────────────────────────────────────────────────────────────────────────
    ap_clients = {ap: 0 for ap in AP_NAMES}
    if ('AP', 'ap_station', 'Station') in g_last.edge_types:
        ei_as = g_last[('AP', 'ap_station', 'Station')].edge_index
        for k in range(ei_as.size(1)):
            ap_clients[IDX2AP_NAME[ei_as[0, k].item()]] += 1

    ap_top10_clients = sorted(
        [{"name": k, "clients": v} for k, v in ap_clients.items()],
        key=lambda x: x["clients"], reverse=True
    )[:10]

    # ──────────────────────────────────────────────────────────────────────────
    # ② AP별 RX/​TX 바이트 집계 (추가)
    # ──────────────────────────────────────────────────────────────────────────
    #※ g_last["AP"].x의 컬럼 인덱스를 실제 RX/TX 위치에 맞춰서 수정하세요.
    rx_idx = 1  # 예시: 0번째가 RX 바이트
    tx_idx = 2  # 예시: 1번째가 TX 바이트

    ap_rx = {
        ap: g_last["AP"].x[i, rx_idx].item()
        for i, ap in enumerate(AP_NAMES)
    }
    ap_tx = {
        ap: g_last["AP"].x[i, tx_idx].item()
        for i, ap in enumerate(AP_NAMES)
    }

    ap_top10_rx = sorted(
        [{"name": k, "rx": v} for k, v in ap_rx.items()],
        key=lambda x: x["rx"], reverse=True
    )[:10]

    ap_top10_tx = sorted(
        [{"name": k, "tx": v} for k, v in ap_tx.items()],
        key=lambda x: x["tx"], reverse=True
    )[:10]

    # ──────────────────────────────────────────────────────────────────────────
    # ③ Station별 연결 AP 수 집계 (기존)
    # ──────────────────────────────────────────────────────────────────────────
    sta_cnt = {ip: 0 for ip in g_last.station_ip2idx.keys()}
    if ('AP', 'ap_station', 'Station') in g_last.edge_types:
        ei_as = g_last[('AP', 'ap_station', 'Station')].edge_index
        for k in range(ei_as.size(1)):
            sta_ip = IDX2STATION_IP.get(ei_as[1, k].item())
            if sta_ip in sta_cnt:
                sta_cnt[sta_ip] += 1

    sta_top10 = sorted(
        [{"ip": k, "connections": v} for k, v in sta_cnt.items()],
        key=lambda x: x["connections"], reverse=True
    )[:10]

    total_sta = g_last["Station"].x.size(0)

    # ──────────────────────────────────────────────────────────────────────────
    # ④ 테이블 생성 헬퍼 함수 (기존)
    # ──────────────────────────────────────────────────────────────────────────
    def simple_table(headers, rows, row_colors=None):
        table_rows = []

        # 헤더
        table_rows.append(
            html.Tr([html.Th(col) for col in headers])
        )

        # 데이터 행
        for row in rows:
            ap_name = row[0]
            row_style = {"backgroundColor": row_colors.get(ap_name, "white")} if row_colors else {}
            table_rows.append(
                html.Tr([html.Td(cell) for cell in row], style=row_style)
            )

        return html.Table(table_rows, style={"width": "100%", "borderCollapse": "collapse"})

    # 색상: 10개 + 기타용 회색
    colors = plotly.colors.qualitative.Plotly[:10] + ["#d9d9d9"]
    top10_ap_names = [d["name"] for d in ap_top10_clients]
    ap_color_map = {name: colors[i] for i, name in enumerate(top10_ap_names)}
    # ──────────────────────────────────────────────────────────────────────────
    # ⑤ 레이아웃 구성
    # ──────────────────────────────────────────────────────────────────────────
    return html.Div(
        [
            # Row 1: AP Up/Down Pie | Total Stations 카드
            html.Div(
                style={"display": "flex", "gap": 16},
                children=[
                    # 1) AP Up/Down Pie Chart
                    html.Div(
                        dcc.Graph(
                            figure=go.Figure(
                                data=[go.Pie(
                                    labels=[d["name"] for d in ap_top10_clients] + ["기타"],
                                    values=[d["clients"] for d in ap_top10_clients] + [
                                        sum(ap_clients.values()) - sum(d["clients"] for d in ap_top10_clients)],
                                    hole=0.4,
                                    marker={"colors": colors},
                                )],
                                layout=go.Layout(
                                    title="Top 10 AP Client 비율",
                                    margin={"l": 10, "r": 10, "t": 30, "b": 10},
                                    height=400,  # ✅ 높이 조절
                                    width=400,  # ✅ 너비 조절 (optional)
                                ),
                            ),
                            config={"displayModeBar": False},
                            style={"height": "400px"},  # ✅ Div 높이도 일치시킴
                        ),
                        style={"width": "30%"},  # ✅ 너비도 넉넉히 조절
                    ),
                    # 2) Total Stations 카드
                    html.Div(
                        [
                            html.Div("Total Stations", style={"fontWeight": "bold"}),
                            html.Div(str(total_sta), style={"fontSize": "32px", "color": "#52c41a"}),
                        ],
                        style={"width": "24%", "textAlign": "center"},
                    ),
                ],
            ),

            # Row 2: AP Top 10 테이블 (Clients, RX, TX), Station Top10 테이블
            html.Div(
                style={"display": "flex", "gap": 16, "marginTop": 32,},
                children=[
                    # A) AP Top 10 (Clients)
                    html.Div(
                        [
                            html.Div("AP Top 10 (Clients)", style={"fontWeight": "bold", "marginBottom": 8}),
                            simple_table(
                                ["AP 이름", "클라이언트 수"],
                                [(d["name"], d["clients"]) for d in ap_top10_clients],

                            ),
                        ],
                        style={"width": "24%"},
                    ),

                    # B) AP Top 10 (RX Bytes)
                    html.Div(
                        [
                            html.Div("AP Top 10 (RX Bytes)", style={"fontWeight": "bold", "marginBottom": 8}),
                            simple_table(
                                ["AP 이름", "RX 바이트"],
                                [(d["name"], f"{d['rx']:,}") for d in ap_top10_rx],


                            ),
                        ],
                        style={"width": "24%"},
                    ),

                    # C) AP Top 10 (TX Bytes)
                    html.Div(
                        [
                            html.Div("AP Top 10 (TX Bytes)", style={"fontWeight": "bold", "marginBottom": 8}),
                            simple_table(
                                ["AP 이름", "TX 바이트"],
                                [(d["name"], f"{d['tx']:,}") for d in ap_top10_tx],


                            ),
                        ],
                        style={"width": "24%"},
                    ),

                    # D) Station Top 10 (연결 AP 수)
                    html.Div(
                        [
                            html.Div("Station Top 10 (연결 AP 수)", style={"fontWeight": "bold", "marginBottom": 8}),
                            simple_table(
                                ["Station IP", "연결 AP 수"],
                                [(d["ip"], d["connections"]) for d in sta_top10]
                            ),
                        ],
                        style={"width": "24%"},
                    ),
                ],
            ),

            # Row 3: AP별 Client Count Bar Chart (기존)
            html.Div(
                style={"marginTop": 32},
                children=[
                    html.Div("AP별 Client Count (Top 10)", style={"fontWeight": "bold", "marginBottom": 8}),
                    dcc.Graph(
                        figure=go.Figure(
                            data=[go.Bar(
                                x=[d["name"] for d in ap_top10_clients],
                                y=[d["clients"] for d in ap_top10_clients],
                                marker={"color": colors},

                            )],
                            layout=go.Layout(
                                xaxis={"title": "AP Name"},
                                yaxis={"title": "client Number"},
                                margin={"l": 40, "r": 10, "t": 20, "b": 40},
                            ),
                        ),
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
            ),

            # Row 4: AP별 RX/​TX Bar Chart (추가)
            html.Div(
                style={"marginTop": 32, "display": "flex", "gap": 16},
                children=[
                    # AP Top 10 (RX Bytes) Bar Chart
                    html.Div(
                        [
                            html.Div("AP별 RX Bytes (Top 10)", style={"fontWeight": "bold", "marginBottom": 8}),
                            dcc.Graph(
                                figure=go.Figure(
                                    data=[go.Bar(
                                        x=[d["name"] for d in ap_top10_rx],
                                        y=[d["rx"] for d in ap_top10_rx],
                                        marker={"color": colors},

                                    )],
                                    layout=go.Layout(
                                        xaxis={"title": "AP Name"},
                                        yaxis={"title": "RX bytes"},
                                        margin={"l": 40, "r": 10, "t": 20, "b": 40},
                                    ),
                                ),
                                config={"displayModeBar": False},
                                style={"height": "300px"},
                            ),
                        ],
                        style={"width": "49%"},
                    ),

                    # AP Top 10 (TX Bytes) Bar Chart
                    html.Div(
                        [
                            html.Div("AP별 TX Bytes (Top 10)", style={"fontWeight": "bold", "marginBottom": 8}),
                            dcc.Graph(
                                figure=go.Figure(
                                    data=[go.Bar(
                                        x=[d["name"] for d in ap_top10_tx],
                                        y=[d["tx"] for d in ap_top10_tx],
                                        marker={"color": colors},

                                    )],
                                    layout=go.Layout(
                                        xaxis={"title": "AP Name"},
                                        yaxis={"title": "TX bytes"},
                                        margin={"l": 40, "r": 10, "t": 20, "b": 40},
                                    ),
                                ),
                                config={"displayModeBar": False},
                                style={"height": "300px"},
                            ),
                        ],
                        style={"width": "49%"},
                    ),
                ],
            ),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. AP 리스트 탭 콘텐츠 (Streaming Mini Chart 포함)
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_list_tab():
    g_last     = dataset[-1]
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
            "status": "UP",
            "channel": int(vec[3]),
            "rx": int(vec[0]),
            "tx": int(vec[1]),
            "clients": ap_clients[ap],
        })

    return html.Div(
        [
            html.Div(
                style={"display": "flex", "gap": 16},
                children=[
                    # Left: AP 정보 DataTable
                    html.Div(
                        dash_table.DataTable(
                            id="ap-table",
                            columns=[
                                {"name": "AP 이름",       "id": "name"},
                                {"name": "MAC",          "id": "mac"},
                                {"name": "상태",          "id": "status"},
                                {"name": "채널",          "id": "channel"},
                                {"name": "Rx",           "id": "rx",      "type": "numeric"},
                                {"name": "Tx",           "id": "tx",      "type": "numeric"},
                                {"name": "클라이언트 수", "id": "clients", "type": "numeric"},
                            ],
                            data=rows,
                            page_size=20,
                            row_selectable="single",
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
                        ),
                        style={"width": "60%"},
                    ),
                    # Right: AP 미니 차트 (Streaming)
                    html.Div(
                        [
                            html.Div("AP 미니 시계열 (실시간)", style={"fontWeight": "bold", "marginBottom": 8}),
                            dcc.Graph(
                                id="ap-mini-chart",
                                figure={
                                    "data": [
                                        {"x": [], "y": [], "name": "Rx", "mode": "lines+markers", "line": {"color": "#1890ff"}},
                                        {"x": [], "y": [], "name": "Tx", "mode": "lines+markers", "line": {"color": "#52c41a"}},
                                    ],
                                    "layout": {
                                        "margin": {"l": 40, "r": 10, "t": 30, "b": 40},
                                        "xaxis": {"title": "Time", "autorange": True},
                                        "yaxis": {"title": "Bytes"},
                                    },
                                },
                                animate=True,
                                style={"height": "300px"},
                            ),
                        ],
                        style={"width": "40%", "borderLeft": "1px solid #ddd", "paddingLeft": 16},
                    ),
                ],
            )
        ]
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3-C. AP 상세 탭 콘텐츠 (기본값: 첫 번째 AP 선택 + 시계열 그래프 포함)
# ──────────────────────────────────────────────────────────────────────────────
def render_ap_detail_tab():
    """
    AP 상세 탭:
      - Dropdown 에서 기본값으로 첫 번째 AP를 미리 선택
      - 선택된 AP의 최근 스냅샷 메타 정보를 표시
      - 최근 20 스냅샷에 대한 Rx/Tx 트래픽과 클라이언트 수 시계열 그래프
      - 90초마다 global-interval에 의해 전체 갱신
    """
    return html.Div(
        [
            html.Div(
                dcc.Dropdown(
                    id="ap-detail-dropdown",
                    options=[{"label": a, "value": a} for a in AP_NAMES],
                    value=AP_NAMES[0],  # 기본값: 첫 번째 AP
                    style={"width": "40%", "marginBottom": 16},
                )
            ),
            html.Div(id="ap-detail-content"),
        ]
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3-D. Station 탭 콘텐츠
# ──────────────────────────────────────────────────────────────────────────────
def render_station_tab():
    """
    Station 탭:
      - 왼쪽: Station 목록 Table
      - 오른쪽: 선택된 Station의 RSSI 시계열(최근 20 스냅샷) + 로밍 히스토리 출력
      - 90초마다 global-interval에 의해 전체 갱신
    """
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
        [
            html.Div(
                style={"display": "flex", "gap": 16},
                children=[
                    # Left: Station 목록 Table
                    html.Div(
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
                            page_size=20,
                            row_selectable="single",
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
                        ),
                        style={"width": "60%"},
                    ),
                    # Right: Station RSSI Chart + 로밍 히스토리
                    html.Div(
                        [
                            html.Div("Station RSSI (최근 20 스냅샷)", style={"fontWeight": "bold", "marginBottom": 8}),
                            dcc.Graph(id="station-traffic-chart", style={"height": "250px"}),
                            html.Div(id="station-roam-history", style={"marginTop": 16}),
                        ],
                        style={"width": "40%", "borderLeft": "1px solid #ddd", "paddingLeft": 16},
                    ),
                ],
            )
        ]
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4. 탭 전환 콜백 (Output 중복 없이 단 하나만)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def tab_router(tab):
    return {
        "overview":  render_overview_tab,
        "ap-list":   render_ap_list_tab,
        "ap-detail": render_ap_detail_tab,
        "station":   render_station_tab,
    }.get(tab, lambda: html.Div("존재하지 않는 탭입니다."))()

# ──────────────────────────────────────────────────────────────────────────────
# 5-1. AP 클릭 시: 미니 차트 초기화 + 시작 인터벌 기록
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "figure"),
    Output("mini-chart-start-interval", "data"),
    Input("ap-table", "selected_rows"),
    State("stream-interval", "n_intervals"),
)
def reset_mini_chart_and_store_start(rows, current_interval):
    """
    - AP를 클릭하면:
      1) 미니 차트를 빈 Figure 로 초기화
      2) 'stream-interval'이 현재 몇 번째 실행됐는지(n_intervals)를
         mini-chart-start-interval Store에 저장
    """
    if not rows or len(rows) == 0:
        raise PreventUpdate

    # ① 미니 차트를 빈 상태로 초기화
    fig = go.Figure(
        [
            go.Scatter(x=[], y=[], name="Rx", mode="lines+markers", line={"color": "#1890ff"}),
            go.Scatter(x=[], y=[], name="Tx", mode="lines+markers", line={"color": "#52c41a"}),
        ],
        layout=dict(
            margin={"l": 40, "r": 10, "t": 30, "b": 40},
            xaxis={"title": "Time", "autorange": True},
            yaxis={"title": "Bytes"},
        ),
    )

    # ② AP 선택 시점의 n_intervals 값을 Store에 저장
    start_interval = current_interval if current_interval is not None else 0

    return fig, start_interval

# ──────────────────────────────────────────────────────────────────────────────
# 5-2. AP 미니 차트 스트리밍 (1초마다 extendData)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-mini-chart", "extendData"),
    Input("stream-interval", "n_intervals"),
    State("ap-table", "selected_rows"),
    State("mini-chart-start-interval", "data"),
    prevent_initial_call=True,
)
def stream_mini_point(n_intervals, rows, stored_start):
    """
    - 매초 호출 → extendData 로 차트에 새로운 (ts, rx, tx) 포인트 추가
    - stored_start: AP를 선택했을 때의 n_intervals 값
    - next_idx = (현재 n_intervals) - (AP 선택 시의 n_intervals)
    - next_idx가 0,1,2,... 로 증가하면서 TS_LIST 순서대로 값을 보냄
    - next_idx >= N_SNAP 이면 더 이상 데이터 없음 → PreventUpdate
    """
    if not rows or len(rows) == 0 or stored_start is None:
        raise PreventUpdate

    next_idx = n_intervals - stored_start
    if next_idx < 0 or next_idx >= N_SNAP:
        raise PreventUpdate

    g_next     = dataset[next_idx]
    ts_next    = TS_LIST[next_idx].to_pydatetime()
    ap_row_idx = rows[0]

    rx_val = g_next["AP"].x[ap_row_idx][0].item()
    tx_val = g_next["AP"].x[ap_row_idx][1].item()

    return (
        {"x": [[ts_next], [ts_next]], "y": [[rx_val], [tx_val]]},
        [0, 1],
        60
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6-A. AP 상세 탭 콜백 (90초마다 갱신)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("ap-detail-content", "children"),
    Input("ap-detail-dropdown", "value"),
    Input("global-interval", "n_intervals"),
)
def ap_detail(ap_name, _):
    """
    - 드롭다운에서 선택된 AP의 메타 정보를 테이블 형태로 표시
    - 최근 20 스냅샷에 대한 Rx/Tx 트래픽 시계열과 클라이언트 수 시계열 그래프 추가
    - 90초마다 global-interval에 의해 전체 갱신
    """
    if not ap_name:
        return html.Div("AP를 선택하세요.", style={"color": "#888"})

    idx    = AP_NAME2IDX[ap_name]
    g_last = dataset[-1]
    vec    = g_last["AP"].x[idx].cpu().numpy().tolist()
    rx0, tx0, chan, x_c, y_c = vec[0], vec[1], int(vec[3]), vec[-2], vec[-1]

    # 1) Metadata 테이블 생성
    tbl = html.Table(
        [
            html.Tr([html.Th("AP"),      html.Td(ap_name)]),
            html.Tr([html.Th("MAC"),     html.Td(g_last["AP"].meta[idx].get("mac", ""))]),
            html.Tr([html.Th("채널"),     html.Td(chan)]),
            html.Tr([html.Th("x_coord"), html.Td(f"{x_c:.2f}")]),
            html.Tr([html.Th("y_coord"), html.Td(f"{y_c:.2f}")]),
            html.Tr([html.Th("최근 Rx"),  html.Td(int(rx0))]),
            html.Tr([html.Th("최근 Tx"),  html.Td(int(tx0))]),
        ],
        style={"border": "1px solid #ccc", "borderCollapse": "collapse", "width": "100%", "marginBottom": "20px"},
    )

    # 2) 최근 20 스냅샷 범위: 인덱스 구하기
    start_idx = max(0, N_SNAP - 20)
    window    = range(start_idx, N_SNAP)

    times         = []
    rx_vals       = []
    tx_vals       = []
    client_counts = []

    for i in window:
        g = dataset[i]
        times.append(TS_LIST[i].to_pydatetime())
        # Rx, Tx
        rx_vals.append(g["AP"].x[idx][0].item())
        tx_vals.append(g["AP"].x[idx][1].item())
        # 클라이언트 수 계산: ('AP','ap_station','Station') 엣지 검색
        cnt = 0
        if ('AP', 'ap_station', 'Station') in g.edge_types:
            ei_as = g[('AP', 'ap_station', 'Station')].edge_index
            for k in range(ei_as.size(1)):
                if ei_as[0, k].item() == idx:
                    cnt += 1
        client_counts.append(cnt)

    # 3) 트래픽 시계열 그래프 (Rx/Tx)
    traffic_fig = go.Figure(
        data=[
            go.Scatter(x=times, y=rx_vals, name="Rx", mode="lines+markers", line={"color": "#1890ff"}),
            go.Scatter(x=times, y=tx_vals, name="Tx", mode="lines+markers", line={"color": "#52c41a"}),
        ]
    ).update_layout(
        title="최근 20 스냅샷 Rx/Tx 트래픽",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
        xaxis={"title": "Time"},
        yaxis={"title": "Bytes"},
    )

    # 4) 클라이언트 수 시계열 그래프
    client_fig = go.Figure(
        data=[
            go.Bar(x=times, y=client_counts, name="Client Count", marker={"color": "#fa8c16"})
        ]
    ).update_layout(
        title="최근 20 스냅샷 클라이언트 수",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
        xaxis={"title": "Time"},
        yaxis={"title": "Clients"},
    )

    return html.Div(
        [
            # Metadata 테이블
            tbl,
            # Rx/Tx 트래픽 그래프
            dcc.Graph(figure=traffic_fig, style={"height": "300px", "marginBottom": "20px"}),
            # 클라이언트 수 그래프
            dcc.Graph(figure=client_fig, style={"height": "300px"}),
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
    """
    - 왼쪽 Station 테이블에서 선택된 Station IP의 RSSI 시계열(최근 20 스냅샷) 그리기
    - 로밍 히스토리가 없으므로 안내 메시지만 표시
    - 90초마다 global-interval에 의해 전체 갱신
    """
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
# 7. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8051)
