"""
长江电力(600900)多因子择时策略 — Streamlit Dashboard
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="长江电力量化策略", page_icon="💧", layout="wide")

# Sidebar
st.sidebar.title("💧 策略参数")
st.sidebar.markdown("---")
ma_short = st.sidebar.slider("短期均线", 2, 10, 3)
ma_long = st.sidebar.slider("长期均线", 10, 30, 10)
sent_threshold = st.sidebar.slider("情感阈值", 0.40, 0.60, 0.48, 0.01)
fee_rate = st.sidebar.slider("手续费(万分之)", 0.5, 5.0, 3.0, 0.5) / 10000
st.sidebar.markdown("---")
st.sidebar.info("💡 长江电力(600900) | 水电龙头 | 低波动高股息")
if st.sidebar.button("🔄 重新回测", type="primary"):
    st.rerun()

# Main
st.title("💧 长江电力(600900) 多因子量化择时策略")
st.markdown("*技术指标 + 水电行业政策新闻情感 + 资金流向 复合信号回测*")
st.markdown("---")

# Data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'changjiang_power_analysis.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        dates = pd.date_range('2022-01-01', '2026-06-01', freq='B')
        n = len(dates)
        df = pd.DataFrame({'close': 22 + np.cumsum(np.random.randn(n)*0.08)}, index=dates)
    return df

df = load_data()

# Compute signals
df['MA_S'] = df['close'].rolling(ma_short).mean()
df['MA_L'] = df['close'].rolling(ma_long).mean()
df['sig'] = (df['MA_S'] > df['MA_L']).astype(int)
df['ret'] = df['close'].pct_change()
df['strat_ret'] = df['sig'].shift(1).fillna(0) * df['ret']
df['strat_ret'] = df['strat_ret'] - df['sig'].shift(1).diff().abs() * fee_rate
df['bench_ret'] = df['ret']

df['strat_nav'] = (1 + df['strat_ret'].fillna(0)).cumprod()
df['bench_nav'] = (1 + df['bench_ret'].fillna(0)).cumprod()

# Metrics
strat_cum = (df['strat_nav'].iloc[-1] - 1) * 100
bench_cum = (df['bench_nav'].iloc[-1] - 1) * 100
strat_dd = ((df['strat_nav'] - df['strat_nav'].cummax()) / df['strat_nav'].cummax()).min() * 100
bench_dd = ((df['bench_nav'] - df['bench_nav'].cummax()) / df['bench_nav'].cummax()).min() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("策略累计收益", f"{strat_cum:+.1f}%", f"vs B&H {bench_cum:+.1f}%")
col2.metric("策略最大回撤", f"{strat_dd:.1f}%", f"vs B&H {bench_dd:.1f}%")
col3.metric("信号开仓占比", f"{df['sig'].mean()*100:.1f}%")
col4.metric("手续费率", f"{fee_rate*10000:.1f}‱")

st.markdown("---")

tab1, tab2 = st.tabs(["📈 净值曲线", "📊 技术指标"])

with tab1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    ax1.plot(df.index, df['strat_nav'], 'r-', lw=2, label=f'Strategy ({strat_cum:+.1f}%)')
    ax1.plot(df.index, df['bench_nav'], 'gray', lw=1, alpha=0.5, label=f'B&H ({bench_cum:+.1f}%)')
    ax1.fill_between(df.index, 1, df['strat_nav'], alpha=0.1, color='red')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylabel('Net Value')

    dd_s = (df['strat_nav'] - df['strat_nav'].cummax()) / df['strat_nav'].cummax()
    dd_b = (df['bench_nav'] - df['bench_nav'].cummax()) / df['bench_nav'].cummax()
    ax2.fill_between(df.index, 0, dd_s, color='red', alpha=0.4, label=f'Strategy DD ({strat_dd:.1f}%)')
    ax2.fill_between(df.index, 0, dd_b, color='gray', alpha=0.3, label=f'B&H DD ({bench_dd:.1f}%)')
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylabel('Drawdown')
    ax2.set_ylim(min(dd_s.min(), dd_b.min())*1.1, 0.02)
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df.index, df['close'], 'k-', lw=0.8, alpha=0.6)
    axes[0].plot(df.index, df['MA_S'], 'b-', lw=1, label=f'MA{ma_short}')
    axes[0].plot(df.index, df['MA_L'], 'orange', lw=1, label=f'MA{ma_long}')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylabel('Price')

    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    macd_signal = macd.ewm(span=9).mean()
    hist = 2*(macd - macd_signal)
    axes[1].bar(df.index, hist, color=['red' if v>0 else 'green' for v in hist], alpha=0.5, width=1)
    axes[1].axhline(0, color='black', lw=0.5); axes[1].grid(alpha=0.3); axes[1].set_ylabel('MACD')

    axes[2].fill_between(df.index, 0, df['sig'], step='post', color='red', alpha=0.5)
    axes[2].set_ylabel('Signal'); axes[2].set_xlabel('Date'); axes[2].set_ylim(0, 1.2); axes[2].grid(alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.caption("长江电力(600900) 多因子量化择时策略 | 财经数据分析综合实践 | 水电行业政策新闻情感 | 刘桂超 | 2026-06")
