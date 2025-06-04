# -*- coding: utf-8 -*-
"""
AGV 전용 실시간 스트리밍 대시보드 예제
- 왼쪽: AGV 목록 테이블 (CSV 마지막 스냅샷 값)
- 오른쪽: 선택된 AGV의 지표(Latency, RSSI, Bitrate, Tx_Bytes)를 1초마다 스트리밍 플로팅
"""
import os
import pandas as pd
import dash
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# 1. 전역 설정
# ──────────────────────────────────────────────────────────────────────────────

# AGV CSV들이 저장된 디렉터리. 예: "./data/agvs/"
PATH_AGV_PREFIX = "data/agvs/"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dash 앱 초기화 및 레이아웃
# ──────────────────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        html.H2("AGV 전용 실시간 스트리밍 대시보드", style={"marginBottom": "20px"}),

        # 1) 1초 주기 인터벌: 스트리밍 콜백을 위해 사용
        dcc.Interval(id="agv-interval", interval=2_000, n_intervals=0),

        # 2) Store: AGV를 선택한 시점의 Interval 카운트를 저장
        dcc.Store(id="agv-start-interval", data=None),
        # 3) Store: 선택된 AGV의 CSV 전체 데이터를 JSON-직렬화 가능한 형태로 저장
        dcc.Store(id="agv-csv-data", data=None),

        # 4) 주요 UI: 왼쪽 AGV 테이블, 오른쪽 스트리밍 그래프 + 상태 메시지
        html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                # ─── 왼쪽: AGV 목록 테이블 ───────────────────────────────────────────────
                html.Div(
                    dash_table.DataTable(
                        id="agv-table",
                        columns=[
                            {"name": "AGV ID",        "id": "agv_id"},
                            {"name": "Connected AP",  "id": "connected_ap"},
                            {"name": "RSSI",          "id": "rssi"},
                            {"name": "Bitrate",       "id": "bitrate"},
                            {"name": "Latency",       "id": "latency"},
                            {"name": "Tx_Bytes",      "id": "tx_bytes"},
                        ],
                        data=[],          # 콜백으로 채워넣을 예정
                        page_size=20,
                        row_selectable="single",
                        style_table={"overflowX": "auto", "height": "700px"},
                        style_cell={"textAlign": "center", "padding": "8px"},
                        style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
                    ),
                    style={"width": "40%"},
                ),

                # ─── 오른쪽: 스트리밍 그래프 + 상태 메시지 ─────────────────────────────────
                html.Div(
                    children=[
                        html.Div("AGV Metrics 실시간 스트리밍", style={"fontWeight": "bold", "marginBottom": "8px"}),
                        dcc.Graph(
                            id="agv-stream-chart",
                            figure={
                                "data": [
                                    {"x": [], "y": [], "name": "Latency",  "mode": "lines+markers", "line": {"color": "#d62728"}},
                                    {"x": [], "y": [], "name": "RSSI",     "mode": "lines+markers", "line": {"color": "#1f77b4"}},
                                    {"x": [], "y": [], "name": "Bitrate",  "mode": "lines+markers", "line": {"color": "#2ca02c"}},
                                    {"x": [], "y": [], "name": "Tx_Bytes","mode": "lines+markers", "line": {"color": "#ff7f0e"}},
                                ],
                                "layout": {
                                    "margin": {"l": 40, "r": 10, "t": 40, "b": 40},
                                    "xaxis": {"title": "Time", "autorange": True},
                                    "yaxis": {"title": "값(단위)"},
                                    "legend": {"orientation": "h", "y": -0.2},
                                },
                            },
                            animate=True,
                            style={"height": "600px"},
                        ),
                        html.Div(id="agv-stream-status", style={"marginTop": "12px", "fontStyle": "italic"}),
                    ],
                    style={"width": "60%", "borderLeft": "1px solid #ddd", "paddingLeft": "16px"},
                ),
            ],
        ),
    ],
    style={"padding": "20px"},
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. 콜백: 앱이 실행되면 처음 AGV 목록을 테이블에 로드
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("agv-table", "data"),
    Input("agv-interval", "n_intervals"),  # 앱 구동 시 한번만 실행돼도 무방
)
def load_agv_list(_):
    """
    - PATH_AGV_PREFIX 폴더 내의 모든 CSV 파일명을 읽어서 AGV ID 목록을 만든 뒤,
      각 CSV의 마지막 행 데이터를 가져와 테이블 데이터에 삽입한다.
    """
    agv_files = [
        f for f in os.listdir(PATH_AGV_PREFIX)
        if os.path.isfile(os.path.join(PATH_AGV_PREFIX, f)) and f.lower().endswith(".csv")
    ]
    agv_ids = sorted(os.path.splitext(f)[0] for f in agv_files)

    rows = []
    for agv_id in agv_ids:
        csv_path = os.path.join(PATH_AGV_PREFIX, f"{agv_id}.csv")
        try:
            df = pd.read_csv(csv_path)
            if df.shape[0] == 0:
                raise ValueError("빈 CSV")

            last = df.iloc[-1]

            # 컬럼 이름 자동 생성
            latency_col   = [c for c in df.columns if 'latency' in c][0]
            rssi_col      = [c for c in df.columns if 'rssi' in c][0]
            bitrate_col   = [c for c in df.columns if 'bitrate' in c][0]
            tx_bytes_col  = [c for c in df.columns if 'tx_bytes' in c][0]
            ap_col        = [c for c in df.columns if 'ap_number' in c][0]

            rows.append({
                "agv_id":       agv_id,
                "connected_ap": last.get(ap_col,    ""),
                "rssi":         last.get(rssi_col,  ""),
                "bitrate":      last.get(bitrate_col,""),
                "latency":      last.get(latency_col,""),
                "tx_bytes":     last.get(tx_bytes_col,""),
            })
        except Exception:
            rows.append({
                "agv_id":       agv_id,
                "connected_ap": "",
                "rssi":         "",
                "bitrate":      "",
                "latency":      "",
                "tx_bytes":     "",
            })

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# 4. 콜백: AGV 선택 + 스트리밍을 모두 처리
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("agv-stream-chart", "figure"),
    Output("agv-stream-chart", "extendData"),
    Output("agv-start-interval", "data"),
    Output("agv-csv-data", "data"),
    Output("agv-stream-status", "children"),
    Input("agv-table",        "selected_rows"),
    Input("agv-interval",     "n_intervals"),
    State("agv-start-interval","data"),
    State("agv-csv-data",     "data"),
    prevent_initial_call=True,
)
def combined_agv_callback(selected_rows, current_interval, stored_start, csv_records):
    """
    - 왼쪽 테이블에서 AGV를 클릭(trigger_id == "agv-table"): CSV를 읽고,
      빈 그래프(Figure)로 초기화하여 반환 + Interval 시작 시점을 기록 + CSV 전체를 Store에 저장 + 상태 메시지.
    - 매초(trigger_id == "agv-interval"):
      stored_start와 csv_records가 존재하면 next_idx 계산 →
      그 인덱스의 레코드를 그래프에 extendData 형태로 추가 + 상태 메시지,
      범위 벗어나면 “스트리밍 완료” 메시지만 출력.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # ──────────────────────────────────────────────────────────────────────────
    # (1) AGV 테이블에서 새로운 행을 클릭했을 때
    # ──────────────────────────────────────────────────────────────────────────
    if trigger_id == "agv-table":
        if not selected_rows:
            raise PreventUpdate

        # 왼쪽 테이블과 동일한 기준으로 AGV ID 목록 생성
        agv_files = [
            f for f in os.listdir(PATH_AGV_PREFIX)
            if os.path.isfile(os.path.join(PATH_AGV_PREFIX, f)) and f.lower().endswith(".csv")
        ]
        agv_ids = sorted(os.path.splitext(f)[0] for f in agv_files)

        sel_idx = selected_rows[0]
        if sel_idx >= len(agv_ids):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div("잘못된 AGV 선택입니다.", style={"color": "red"})

        agv_id   = agv_ids[sel_idx]
        csv_path = os.path.join(PATH_AGV_PREFIX, f"{agv_id}.csv")
        if not os.path.isfile(csv_path):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div(f"CSV 파일이 없습니다: {agv_id}", style={"color": "red"})

        # CSV 전체를 읽어서 DataFrame 생성
        df = pd.read_csv(csv_path)
        if "date_agv2" not in df.columns or "time" not in df.columns:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div("CSV에 'date_agv2' 또는 'time' 컬럼이 없습니다.", style={"color": "red"})

        # timestamp 컬럼 생성
        df["timestamp"] = pd.to_datetime(df["date_agv2"].astype(str) + " " + df["time"].astype(str))
        latency_col = [c for c in df.columns if 'latency' in c][0]
        rssi_col = [c for c in df.columns if 'rssi' in c][0]
        bitrate_col = [c for c in df.columns if 'bitrate' in c][0]
        tx_bytes_col = [c for c in df.columns if 'tx_bytes' in c][0]
        ap_col = [c for c in df.columns if 'ap_number' in c][0]

        # JSON-호환 형태로 전체 레코드를 리스트-오브-딕트로 변환
        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp": row["timestamp"].isoformat(),
                "latency":   row.get(latency_col,   None),
                "rssi":      row.get(rssi_col,      None),
                "bitrate":   row.get(bitrate_col,   None),
                "tx_bytes":  row.get(tx_bytes_col,  None),
                "ap_number": row.get(ap_col,        None),
            })

        # 빈 Figure로 초기화 (4개 trace: Latency, RSSI, Bitrate, Tx_Bytes)
        empty_fig = go.Figure(
            data=[
                {"x": [], "y": [], "name": "Latency",  "mode": "lines+markers", "line": {"color": "#d62728"}},
                {"x": [], "y": [], "name": "RSSI",     "mode": "lines+markers", "line": {"color": "#1f77b4"}},
                {"x": [], "y": [], "name": "Bitrate",  "mode": "lines+markers", "line": {"color": "#2ca02c"}},
                {"x": [], "y": [], "name": "Tx_Bytes","mode": "lines+markers", "line": {"color": "#ff7f0e"}},
            ],
            layout=dict(
                margin={"l": 40, "r": 10, "t": 40, "b": 40},
                xaxis={"title": "Time", "autorange": True},
                yaxis={"title": "값(단위)"},
                legend={"orientation": "h", "y": -0.2},
            ),
        )

        # 스트리밍 시작 시점(= agv-interval.n_intervals)
        start_interval = current_interval if current_interval is not None else 0

        status_msg = html.Div(f"AGV {agv_id} CSV 로드 완료. 스트리밍을 시작합니다.", style={"color": "#52c41a"})

        # (리턴 순서)
        # 1) figure (빈 차트)
        # 2) extendData (아직 없으므로 no_update)
        # 3) agv-start-interval (start_interval 저장)
        # 4) agv-csv-data (records 저장)
        # 5) 상태 메시지
        return empty_fig, dash.no_update, start_interval, records, status_msg

    # ──────────────────────────────────────────────────────────────────────────
    # (2) 1초마다 agv-interval이 트리거될 때(스트리밍 중)
    # ──────────────────────────────────────────────────────────────────────────
    elif trigger_id == "agv-interval":
        # (아직 AGV를 선택하지 않아서 데이터가 없으면 업데이트 금지)
        if stored_start is None or csv_records is None:
            raise PreventUpdate

        next_idx = current_interval - stored_start
        total_len = len(csv_records)

        # (범위를 벗어나면 스트리밍 완료)
        if next_idx < 0 or next_idx >= total_len:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div("스트리밍 완료", style={"color": "#888"})

        rec = csv_records[next_idx]

        # ISO 문자열을 다시 pandas.Timestamp로 변환
        ts = pd.to_datetime(rec["timestamp"])

        latency_val   = rec.get("latency",   None) or 0
        rssi_val      = rec.get("rssi",      None) or 0
        bitrate_val   = rec.get("bitrate",   None) or 0
        tx_bytes_val  = rec.get("tx_bytes",  None) or 0

        extend_dict = {
            "x": [[ts], [ts], [ts], [ts]],
            "y": [[latency_val], [rssi_val], [bitrate_val], [tx_bytes_val]],
        }
        trace_indices = [0, 1, 2, 3]

        status_msg = html.Div(f"Streaming: {next_idx+1}/{total_len}", style={"color": "#1f77b4"})

        # (리턴 순서)
        # 1) figure: no_update (이미 빈 차트가 그려져 있음)
        # 2) extendData: 새로운 데이터 포인트
        # 3) agv-start-interval: no_update (이미 기록된 값을 유지)
        # 4) agv-csv-data: no_update (이미 기록된 전체 CSV 유지)
        # 5) 상태 메시지
        return dash.no_update, (extend_dict, trace_indices), dash.no_update, dash.no_update, status_msg

    # ──────────────────────────────────────────────────────────────────────────
    # (3) 위 두 경우 외에는 아무 것도 하지 않음
    # ──────────────────────────────────────────────────────────────────────────
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# ──────────────────────────────────────────────────────────────────────────────
# 5. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

