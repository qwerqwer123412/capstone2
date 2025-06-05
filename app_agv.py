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
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

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
        html.H2("AGV 전용 실시간 업데이트 대시보드", style={"marginBottom": "20px"}),

        # 1초마다 호출될 인터벌
        dcc.Interval(id="agv-interval", interval=1_000, n_intervals=0),

        # 테이블과 차트(선택된 AGV), 하지만 여기서는 테이블만 예시로 업데이트
        html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                # 왼쪽: AGV 목록 테이블 (매초 값 갱신)
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
                        style_table={"overflowX": "auto", "height": "700px"},
                        style_cell={"textAlign": "center", "padding": "8px"},
                        style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
                    ),
                    style={"width": "100%"},
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

        rows.append({
            "agv_id":       agv_id,
            "connected_ap": ap_val,
            "rssi":         rssi_val,
            "bitrate":      bitrate_val,
            "latency":      latency_val,
            "tx_bytes":     tx_bytes_val,
        })

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# 4. 앱 실행
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
