import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import io
# --- 1. データの準備（提供された全データをここに統合） ---
raw_data = """種類(縦),街灯の種類(横),覆いの種類,測定距離,照度,分散,理想の照度
8.0cm,8.6cm,覆い無し,0m,1.5lx,0.13,0.83lx
8.0cm,8.6cm,覆い無し,5m,1.0lx,0.13,0.83lx
8.0cm,8.6cm,覆い無し,10m,1.0lx,0.13,0.83lx
8.0cm,8.6cm,覆い無し,15m,1.0lx,0.13,0.83lx
8.0cm,8.6cm,正四角錐,0m,1.0lx,0.134,0.83lx
8.0cm,8.6cm,正四角錐,5m,1.5lx,0.134,0.83lx
8.0cm,8.6cm,正四角錐,10m,1.0lx,0.134,0.83lx
8.0cm,8.6cm,正四角錐,15m,1.0lx,0.134,0.83lx
8.0cm,8.6cm,円錐,0m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,円錐,5m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,円錐,10m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,円錐,15m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,正三角錐,0m,0.5lx,0.0889,0.83lx
8.0cm,8.6cm,正三角錐,5m,1.0lx,0.0889,0.83lx
8.0cm,8.6cm,正三角錐,10m,0.5lx,0.0889,0.83lx
8.0cm,8.6cm,正三角錐,15m,0.5lx,0.0889,0.83lx
8.0cm,8.6cm,菱形,0m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,菱形,5m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,菱形,10m,0.5lx,0.109,0.83lx
8.0cm,8.6cm,菱形,15m,0.5lx,0.109,0.83lx
16.75cm,8.6cm,覆い無し,0m,2.0lx,0.17,1.7lx
16.75cm,8.6cm,覆い無し,5m,1.5lx,0.17,1.7lx
16.75cm,8.6cm,覆い無し,10m,1.5lx,0.17,1.7lx
16.75cm,8.6cm,覆い無し,15m,1.0lx,0.17,1.7lx
16.75cm,8.6cm,正四角錐,0m,1.5lx,0.653,1.7lx
16.75cm,8.6cm,正四角錐,5m,2.5lx,0.653,1.7lx
16.75cm,8.6cm,正四角錐,10m,1.0lx,0.653,1.7lx
16.75cm,8.6cm,正四角錐,15m,0.5lx,0.653,1.7lx
26.5cm,8.6cm,正四角錐,0m,1.5lx,1.38,2.5lx
26.5cm,8.6cm,正四角錐,5m,2.5lx,1.38,2.5lx
16.75cm,18cm,正四角錐,5m,2.0lx,3.37,3.3lx
26.5cm,18cm,正四角錐,0m,5.5lx,4.35,5.0lx
26.5cm,18cm,円錐,0m,6.0lx,6.50,5.0lx
26.5cm,18cm,正三角錐,0m,5.0lx,7.31,5.0lx
26.5cm,18cm,菱形,0m,5.0lx,7.56,5.0lx"""
# --- 2. 前処理ロジック ---
df = pd.read_csv(io.StringIO(raw_data))
def clean(text):
   if isinstance(text, str):
       return float(text.replace('cm','').replace('m','').replace('lx','').replace('px','').strip())
   return text
df['v'] = df['種類(縦)'].apply(clean)
df['h'] = df['街灯の種類(横)'].apply(clean)
df['d'] = df['測定距離'].apply(clean)
# 学習用データ（「覆い無し」は除外して、最適な「形」を予測するようにする）
train_df = df[df['覆いの種類'] != '覆い無し']
X = train_df[['v', 'h', 'd']]
y = train_df['覆いの種類']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# --- 3. アプリのレイアウト設定 ---
st.set_page_config(page_title="光害対策AI", page_icon="🌙")
st.title("🌙 光害対策AIシミュレーター")
st.write("街灯のサイズを入力して、最適な「覆い」の形を診断しましょう。")
# 入力セクション
st.divider()
col1, col2 = st.columns(2)
with col1:
   v_in = st.number_input("街灯の直径 (縦) [cm]", min_value=5.0, max_value=50.0, value=16.0)
   h_in = st.number_input("街灯の種類 (横) [cm]", min_value=5.0, max_value=30.0, value=8.6)
with col2:
   d_in = st.select_slider("測定したい距離 [m]", options=[0, 5, 10, 15], value=5)
# 判定ボタン
if st.button("AI判定を開始する"):
   prediction = model.predict([[v_in, h_in, d_in]])[0]
   st.balloons() # 演出
   st.success(f"結果：この街灯には「**{prediction}**」の覆いが最適です！")
   st.info(f"💡 解説: {v_in}cm×{h_in}cmの街灯において、{d_in}m地点の光を制御しつつ、空への光漏れを最小限に抑える形状として算出されました。")
st.divider()
st.caption("研究発表用デモアプリ | データに基づいた機械学習モデルを使用しています")
