# -*- coding: utf-8 -*-
"""
AGV 전용 실시간 업데이트 대시보드 예제
- 앱 시작 시 모든 AGV CSV를 전역 변수로 로드 (agv_df_dict)
- 1초마다 dcc.Interval이 트리거되면, 각 AGV별로 현재 인덱스에 해당하는 row를 꺼내 테이블을 갱신
"""
import os
import pandas as pd
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# 1. 전역 변수에 AGV CSV 모두 로드
# ──────────────────────────────────────────────────────────────────────────────

# 실제 AGV CSV들이 저장된 폴더 경로
PATH_AGV_PREFIX = "data/agvs/"

# 전역 딕셔너리: { "132": DataFrame, "133": DataFrame, ... }
# 앱 시작 시 한 번만 모든 CSV를 읽어서 여기에 저장
agv_df_dict = {}

for fname in sorted(os.listdir(PATH_AGV_PREFIX)):
    if not fname.lower().endswith(".csv"):
        continue
    agv_id = os.path.splitext(fname)[0]
    full_path = os.path.join(PATH_AGV_PREFIX, fname)
    try:
        df = pd.read_csv(full_path)
        # 빈 파일 스킵
        if df.empty:
            continue
        # "date_agv2" + "time" → "timestamp" 컬럼 합치기
        if "date_agv2" in df.columns and "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date_agv2"].astype(str) + " " + df["time"].astype(str))
        else:
            # date/time 컬럼이 없으면, 경고 없이 timestamp 없이 넘어감
            df["timestamp"] = pd.NaT

        # 전역 dict에 저장
        agv_df_dict[agv_id] = df

    except Exception:
        # 읽기 오류가 발생하면 해당 AGV는 스킵
        continue

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dash 앱 초기화 및 레이아웃
# ──────────────────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


app.layout = html.Div(
    [
        html.H2("AGV 전용 실시간 대시보드 (테이블 + 30스텝 그래프)", style={"marginBottom": "20px"}),

        # 1초마다 호출될 인터벌
        dcc.Interval(id="agv-interval", interval=200, n_intervals=0),

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
                        data=[],          # 콜백에서 채워 넣을 예정
                        page_size=20,
                        row_selectable="single",
                        style_table={"overflowX": "auto", "height": "700px"},
                        style_cell={"textAlign": "center", "padding": "8px"},
                        style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
                    ),
                    style={"width": "40%"},
                ),

                # ─── 오른쪽: 선택된 AGV 최대 30스텝 그래프 ─────────────────────────────────
                html.Div(
                    children=[
                        html.Div("선택된 AGV 최대 30스텝 지표", style={"fontWeight": "bold", "marginBottom": "8px"}),
                        dcc.Graph(
                            id="agv-graph",
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
                                    "yaxis": {"title": "Value"},
                                    "legend": {"orientation": "h", "y": -0.2},
                                },
                            },
                            animate=False,  # 매초 전체 데이터를 갱신하므로 animate=False
                            style={"height": "600px"},
                        ),
                        html.Div(
                            "테이블에서 AGV를 선택하면 우측 그래프가 30스텝 이전부터 실시간으로 갱신됩니다.",
                            style={"marginTop": "12px", "fontStyle": "italic", "color": "#666"},
                        ),
                    ],
                    style={"width": "60%", "borderLeft": "1px solid #ddd", "paddingLeft": "16px"},
                ),
            ],
        ),
    ],
    style={"padding": "20px"},
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. 콜백: 1초마다 agv-interval → 테이블 데이터 갱신
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("agv-table", "data"),
    Input("agv-interval", "n_intervals"),
)
def update_agv_table(n_intervals):
    """
    - 매초(agv-interval 트리거) 호출
    - 전역 agv_df_dict에 미리 로드해 둔 DataFrame에서
      current_idx = n_intervals 에 해당하는 행(존재하지 않으면 마지막 행)을 꺼내
      각 AGV의 connected_ap, rssi, bitrate, latency, tx_bytes 값을 테이블에 넣는다.
    """
    if not agv_df_dict:
        raise PreventUpdate

    rows = []
    # 모든 AGV ID 순회
    for agv_id, df in agv_df_dict.items():
        # 현재 인덱스 (n_intervals)가 DF 길이를 벗어나면 마지막 행을 사용
        idx = n_intervals
        if idx >= len(df):
            idx = len(df) - 1

        row = df.iloc[idx]
        # 컬럼 이름 자동 탐색
        #   latency_col, rssi_col, bitrate_col, tx_bytes_col, ap_col
        #   (예: "latency_agv2_132", "rssi_agv2_132", ...)
        latency_cols  = [c for c in df.columns if "latency"   in c]
        rssi_cols     = [c for c in df.columns if "rssi"      in c]
        bitrate_cols  = [c for c in df.columns if "bitrate"   in c]
        tx_bytes_cols = [c for c in df.columns if "tx_bytes"  in c]
        ap_cols       = [c for c in df.columns if "ap_number" in c]

        latency_val   = row[latency_cols[0]]  if latency_cols  else ""
        rssi_val      = row[rssi_cols[0]]     if rssi_cols     else ""
        bitrate_val   = row[bitrate_cols[0]]  if bitrate_cols  else ""
        tx_bytes_val  = row[tx_bytes_cols[0]] if tx_bytes_cols else ""
        ap_val        = row[ap_cols[0]]       if ap_cols       else ""

        def fmt(x):
            return f"{x:.2f}" if (x is not None and pd.notna(x)) else ""

        rows.append({
            "agv_id":       agv_id,
            "connected_ap": ap_val,
            "rssi":         rssi_val,
            "bitrate":      bitrate_val,
            "latency":      fmt(latency_val),
            "tx_bytes":     fmt(tx_bytes_val),
        })

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 4. 콜백: 선택된 AGV의 최대 30스텝 그래프 업데이트
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("agv-graph", "figure"),
    Input("agv-interval", "n_intervals"),
    State("agv-table",   "selected_rows"),
)
def update_agv_graph(n_intervals, selected_rows):
    """
    - 매초(agv-interval 트리거) 호출
    - 사용자가 테이블에서 AGV를 클릭(selected_rows)하지 않았다면 빈 그래프 반환
    - selected_rows가 있으면, 해당 AGV ID를 찾고 전역 DataFrame에서
      current_idx = min(n_intervals, len(df)-1)
      start_idx   = max(0, current_idx - 29)
      [start_idx : current_idx+1] 구간을 잘라서 Figure로 생성
    """
    # 1) 아무 AGV도 선택되지 않았다면 → 빈 그래프 반환
    if not selected_rows:
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
        return empty_fig

    # 2) 선택된 AGV ID 추출
    sel_idx = selected_rows[0]
    # 테이블은 update_agv_table 콜백에서 매초 갱신되므로
    # agv_df_dict.keys()의 순서와 동일한 정렬 상태라고 가정
    agv_ids = sorted(agv_df_dict.keys())
    if sel_idx >= len(agv_ids):
        raise PreventUpdate

    agv_id = agv_ids[sel_idx]
    df = agv_df_dict[agv_id]

    # 3) current_idx 계산(범위를 벗어나면 마지막 인덱스 사용)
    current_idx = n_intervals
    if current_idx >= len(df):
        current_idx = len(df) - 1

    # 4) 최대 30스텝 이전(start_idx)부터 current_idx까지 범위를 잘라낸다.
    start_idx = max(0, current_idx - 29)
    window_df = df.iloc[start_idx : current_idx + 1]

    # 5) 각 컬럼 이름 자동 추출
    latency_col   = [c for c in df.columns if "latency"   in c][0]  if [c for c in df.columns if "latency"   in c] else None
    rssi_col      = [c for c in df.columns if "rssi"      in c][0]  if [c for c in df.columns if "rssi"      in c] else None
    bitrate_col   = [c for c in df.columns if "bitrate"   in c][0]  if [c for c in df.columns if "bitrate"   in c] else None
    tx_bytes_col  = [c for c in df.columns if "tx_bytes"  in c][0]  if [c for c in df.columns if "tx_bytes"  in c] else None

    # 6) Figure 생성
    fig = go.Figure()

    # Latency trace (yaxis="y")
    if latency_col:
        fig.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df[latency_col],
            name="Latency", mode="lines+markers", line={"color": "#d62728"}, yaxis="y"
        ))
    # RSSI trace (yaxis="y2")
    if rssi_col:
        fig.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df[rssi_col],
            name="RSSI", mode="lines+markers", line={"color": "#1f77b4"}, yaxis="y2"
        ))
    # Bitrate trace (yaxis="y3")
    if bitrate_col:
        fig.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df[bitrate_col],
            name="Bitrate", mode="lines+markers", line={"color": "#2ca02c"}, yaxis="y3"
        ))
    # Tx_Bytes trace (yaxis="y4")
    if tx_bytes_col:
        fig.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df[tx_bytes_col],
            name="Tx_Bytes", mode="lines+markers", line={"color": "#ff7f0e"}, yaxis="y4"
        ))

    fig.update_layout(
        title=f"AGV {agv_id} 최대 30스텝 실시간 플롯",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        xaxis={"title": "Time", "autorange": True},
        yaxis=dict(
            title="Latency",
            anchor="x",
            side="left",
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showticklabels=False,
            title="",  # 레이블 숨김
        ),
        yaxis3=dict(
            overlaying="y",
            side="right",
            position=0.94,
            showticklabels=False,
            title="",  # 레이블 숨김
        ),
        yaxis4=dict(
            overlaying="y",
            side="right",
            position=0.98,
            showticklabels=False,
            title="",  # 레이블 숨김
        ),
        legend={"orientation": "h", "y": -0.2},
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)