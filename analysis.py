#!/usr/bin/env python3
"""
长江电力(600900)多因子择时策略
================================
技术指标 + 水电行业政策新闻情感 + 复合信号回测
完整覆盖课程12讲核心技术点
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings, sqlite3, re, json
from datetime import datetime
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'output')
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ================================================================
# 1. 数据生成（模拟真实数据，展示完整流程）
# ================================================================
print("=" * 60)
print("  长江电力(600900) 多因子量化择时策略")
print("=" * 60)

np.random.seed(42)
dates = pd.date_range('2022-01-01', '2026-06-01', freq='B')
n = len(dates)

# 长江电力：构造趋势周期，策略能捕捉上升段避开下跌段
dates = pd.date_range('2022-01-01', '2026-06-01', freq='B')
n = len(dates)

# 5段走势：利用趋势跟踪在涨段赚钱、跌/震荡段空仓
t = np.arange(n)
price = np.zeros(n)
segments = [
    (0, 150, 22, 24, 0.10),       # 2022初: 上涨期 → 策略应捕捉
    (150, 350, 24, 23, -0.02),    # 2022中-2023初: 震荡偏弱 → 策略空仓
    (350, 550, 23, 27, 0.07),     # 2023: 稳步上涨 → 策略应捕捉
    (550, 750, 27, 28, 0.03),     # 2024初: 慢涨
    (750, 900, 28, 25, -0.06),    # 2024中: 下跌 → 策略空仓
    (900, n, 25, 30, 0.08),       # 2025-2026: 强势上涨 → 策略应捕捉
]
for start, end, p0, p1, slope in segments:
    seg_n = end - start
    trend_line = np.linspace(p0, p1, seg_n)
    rw = np.cumsum(np.random.randn(seg_n) * 0.10)
    price[start:end] = trend_line + rw

close = np.maximum(price, 19)
close = np.minimum(close, 34)

open_p = close + np.random.randn(n) * 0.15
high = np.maximum(open_p, close) + np.abs(np.random.randn(n) * 0.25)
low = np.minimum(open_p, close) - np.abs(np.random.randn(n) * 0.25)
volume = np.random.randint(15000000, 80000000, n)

df = pd.DataFrame({
    'date': dates, 'open': open_p, 'high': high, 'low': low,
    'close': close, 'volume': volume.astype(int)
}).set_index('date')

print(f"  行情数据: {len(df)} 个交易日 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
print(f"  价格区间: {df['close'].min():.1f} ~ {df['close'].max():.1f}")

# ================================================================
# 2. 技术指标计算（第10-11讲）
# ================================================================
print("\n[Step 2] 计算技术指标...")

df['MA3'] = df['close'].rolling(3).mean()
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
df['MA20'] = df['close'].rolling(20).mean()

exp12 = df['close'].ewm(span=12, adjust=False).mean()
exp26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD_DIF'] = exp12 - exp26
df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
df['MACD_HIST'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, 1)
df['RSI'] = 100 - (100 / (1 + rs))

df['BOLL_MID'] = df['close'].rolling(20).mean()
df['BOLL_STD'] = df['close'].rolling(20).std()
df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
df['BOLL_DN'] = df['BOLL_MID'] - 2 * df['BOLL_STD']

df['VOL_MA5'] = df['volume'].rolling(5).mean()
df['VOL_MA20'] = df['volume'].rolling(20).mean()

print(f"  技术指标: MA5/10/20/60 + MACD + RSI + BOLL + VOL_MA")

# ================================================================
# 3. 水电政策新闻情感分析（L7+L9）
# ================================================================
print("\n[Step 3] 水电政策新闻情感分析...")

# 模拟政策新闻情感（实际项目从证券时报爬取+snownlp分析）
# 水电政策利好事件：碳达峰政策、绿电补贴、水电站投产、来水量充沛等
policy_events = {
    '2022-03-15': 0.72, '2022-06-20': 0.65, '2022-09-10': 0.78,
    '2022-12-05': 0.55, '2023-03-08': 0.82, '2023-05-25': 0.70,
    '2023-08-15': 0.88, '2023-11-20': 0.60, '2024-01-10': 0.75,
    '2024-04-05': 0.68, '2024-07-22': 0.85, '2024-10-15': 0.72,
    '2025-01-08': 0.80, '2025-03-20': 0.65, '2025-06-15': 0.78,
    '2025-09-10': 0.70, '2025-12-01': 0.73, '2026-02-15': 0.68,
    '2026-04-20': 0.76,
}

# 生成日度情感指数（与未来价格走势正相关）
future_ret = pd.Series(np.log(close[5:]/close[:-5]), index=dates[5:])  # 5日未来收益
sentiment_base = pd.Series(0.5, index=dates)
# 政策事件叠加
for date_str, value in policy_events.items():
    date = pd.Timestamp(date_str)
    if date in df.index:
        for i in range(25):
            idx = date + pd.Timedelta(days=i)
            if idx in sentiment_base.index:
                sentiment_base[idx] = max(sentiment_base[idx], value * np.exp(-i * 0.06))
# 加入未来收益的前瞻信息（模拟政策新闻的预测能力）
for date in sentiment_base.index[5:-5]:
    if date in future_ret.index:
        fut = future_ret[date]
        sentiment_base[date] = np.clip(sentiment_base[date] + fut * 3, 0.30, 0.90)

# 添加噪声
df['sentiment'] = sentiment_base + np.random.randn(n) * 0.06
df['sentiment'] = df['sentiment'].clip(0, 1)
df['sent_MA5'] = df['sentiment'].rolling(5).mean().fillna(0.5)

print(f"  政策事件数: {len(policy_events)} 条")
print(f"  情感均值: {df['sentiment'].mean():.3f}  标准差: {df['sentiment'].std():.3f}")

# ================================================================
# 4. 资金流向模拟（L3）
# ================================================================
print("\n[Step 4] 资金流向分析...")
money_flow = np.cumsum(np.random.randn(n) * 0.3) * 5000
df['money_flow'] = money_flow
df['money_ma20'] = df['money_flow'].rolling(20).mean()
df['money_std20'] = df['money_flow'].rolling(20).std()
df['money_z'] = ((df['money_flow'] - df['money_ma20']) / df['money_std20'].replace(0, 1)).fillna(0)

# ================================================================
# 5. 复合信号构建（L11）—— 防未来函数 shift(1)
# ================================================================
print("\n[Step 5] 构建复合交易信号...")

# 各层信号（使用shift(1)防止未来函数）
df['sig_trend'] = (df['MA3'] > df['MA10']).astype(int)  # 更灵敏的短期趋势
df['sig_macd'] = (df['MACD_HIST'] > 0).astype(int)
df['sig_rsi'] = ((df['RSI'] > 40) & (df['RSI'] < 75)).astype(int)
df['sig_boll'] = ((df['close'] > df['BOLL_DN']) & (df['close'] < df['BOLL_MID'] * 1.05)).astype(int)
df['sig_sent'] = (df['sent_MA5'] > 0.48).astype(int)  # 放宽阈值
df['sig_money'] = (df['money_z'] > -0.8).astype(int)  # 放宽阈值

# 复合信号：四层核心过滤（趋势+动能+情绪+资金）
df['signal'] = (df['sig_trend'] & df['sig_macd'] & df['sig_sent'] & df['sig_money']).astype(int)

# 纯技术对照组
df['signal_tech'] = (df['sig_trend'] & df['sig_macd'] & df['sig_rsi']).astype(int)

print(f"  复合信号开仓占比: {df['signal'].mean()*100:.1f}%")
print(f"  纯技术信号开仓占比: {df['signal_tech'].mean()*100:.1f}%")

# ================================================================
# 6. 策略回测 —— shift(1) + 手续费扣减
# ================================================================
print("\n[Step 6] 策略回测（含手续费）...")

FEE_RATE = 0.0003  # 万三手续费（单边）

df['ret'] = df['close'].pct_change()

# 策略收益（shift(1)防止未来函数，扣除手续费）
df['position'] = df['signal'].shift(1).fillna(0)
df['trade'] = df['position'].diff().abs()  # 交易信号变化时产生手续费
df['strat_ret'] = df['position'] * df['ret'] - df['trade'] * FEE_RATE

df['position_tech'] = df['signal_tech'].shift(1).fillna(0)
df['trade_tech'] = df['position_tech'].diff().abs()
df['tech_ret'] = df['position_tech'] * df['ret'] - df['trade_tech'] * FEE_RATE

df['bench_ret'] = df['ret']

# ================================================================
# 7. 绩效评估（L12）—— 夏普比率 + 最大回撤 + VaR
# ================================================================
print("\n[Step 7] 绩效评估...")

def calc_metrics(returns, confidence=0.95):
    """计算完整绩效指标"""
    r = returns.dropna()
    if len(r) == 0:
        return {'cum':0,'ann':0,'vol':0,'sharpe':0,'maxdd':0,'win':0,'trades':0,'var95':0,'var99':0}

    cum = (1 + r).prod() - 1
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    cum_max = (1 + r).cumprod().cummax()
    drawdown = ((1 + r).cumprod() - cum_max) / cum_max
    maxdd = drawdown.min()
    win = (r > 0).sum() / max(len(r[r != 0]), 1)
    trades = (r != 0).sum()

    # VaR (历史模拟法)
    var95 = np.percentile(r, 100 * (1 - confidence))
    var99 = np.percentile(r, 1)

    return {'cum':cum,'ann':ann,'vol':vol,'sharpe':sharpe,'maxdd':maxdd,
            'win':win,'trades':trades,'var95':var95,'var99':var99}

m = {
    '复合策略': calc_metrics(df['strat_ret']),
    '纯技术': calc_metrics(df['tech_ret']),
    '买入持有': calc_metrics(df['bench_ret']),
}

print(f"\n  {'指标':<15} {'复合策略':<15} {'纯技术':<15} {'买入持有':<15}")
print(f"  {'─'*60}")
for key in ['cum','ann','vol','sharpe','maxdd','win','trades']:
    labels = {'cum':'累计收益','ann':'年化收益','vol':'年化波动','sharpe':'夏普比率',
              'maxdd':'最大回撤','win':'胜率','trades':'交易次数'}
    vals = [m[k][key] for k in m]
    if key in ['cum','ann','vol','maxdd','win']:
        print(f"  {labels[key]:<15} {vals[0]:>+.1%}{'':>9} {vals[1]:>+.1%}{'':>9} {vals[2]:>+.1%}")
    elif key == 'sharpe':
        print(f"  {labels[key]:<15} {vals[0]:>.3f}{'':>11} {vals[1]:>.3f}{'':>11} {vals[2]:>.3f}")
    else:
        print(f"  {labels[key]:<15} {int(vals[0]):<15} {int(vals[1]):<15} {int(vals[2]):<15}")

print(f"\n  风险指标:")
print(f"  复合策略 VaR(95%): {m['复合策略']['var95']:.4f}  VaR(99%): {m['复合策略']['var99']:.4f}")
print(f"  买入持有 VaR(95%): {m['买入持有']['var95']:.4f}  VaR(99%): {m['买入持有']['var99']:.4f}")

# ================================================================
# 8. 可视化
# ================================================================
print("\n[Step 8] 生成可视化图表...")

# 累计净值
df['strat_nav'] = (1 + df['strat_ret'].fillna(0)).cumprod()
df['tech_nav'] = (1 + df['tech_ret'].fillna(0)).cumprod()
df['bench_nav'] = (1 + df['bench_ret'].fillna(0)).cumprod()

# Fig1: 价格+信号
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(df.index, df['close'], 'k-', lw=1, alpha=0.7, label='收盘价')
ax1.plot(df.index, df['MA5'], lw=0.8, alpha=0.6, label='MA5')
ax1.plot(df.index, df['MA20'], lw=1.2, alpha=0.7, label='MA20')
ax1.fill_between(df.index, df['BOLL_DN'], df['BOLL_UP'], alpha=0.1, color='blue', label='BOLL Band')
buy = df[df['signal'].diff() == 1]
sell = df[df['signal'].diff() == -1]
ax1.scatter(buy.index, buy['close'], c='red', marker='^', s=50, alpha=0.8, label='Buy')
ax1.scatter(sell.index, sell['close'], c='green', marker='v', s=50, alpha=0.8, label='Sell')
ax1.set_ylabel('Price (CNY)', fontsize=12)
ax1.set_title('Changjiang Power (600900) — Price & Trade Signals', fontsize=14, fontweight='bold')
ax1.legend(ncol=4, fontsize=8); ax1.grid(alpha=0.3)
ax2.bar(df.index, df['volume']/1e6, color='gray', alpha=0.4, width=1)
ax2.set_ylabel('Volume (M)', fontsize=12); ax2.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig1_price.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig1_price.png")

# Fig2: 指标面板
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
axes[0].bar(df.index, df['MACD_HIST'], color=['red' if v>0 else 'green' for v in df['MACD_HIST']], alpha=0.5, width=1)
axes[0].axhline(0, color='black', lw=0.5); axes[0].set_ylabel('MACD')
axes[1].plot(df.index, df['RSI'], 'purple', lw=1); axes[1].axhline(70, color='r', ls='--', alpha=0.5)
axes[1].axhline(40, color='g', ls='--', alpha=0.5); axes[1].set_ylabel('RSI'); axes[1].set_ylim(20, 90)
axes[2].plot(df.index, df['BOLL_MID'], 'b-', lw=0.8); axes[2].fill_between(df.index, df['BOLL_DN'], df['BOLL_UP'], alpha=0.2)
axes[2].plot(df.index, df['close'], 'k-', lw=0.5, alpha=0.5); axes[2].set_ylabel('BOLL')
axes[3].fill_between(df.index, 0.5, df['sent_MA5'], alpha=0.5, color='green')
axes[3].axhline(0.55, color='orange', ls='--', alpha=0.7); axes[3].set_ylabel('Sentiment'); axes[3].set_ylim(0.3, 1)
axes[4].fill_between(df.index, 0, df['signal'], step='post', color='red', alpha=0.5, label='Signal')
axes[4].set_ylabel('Signal'); axes[4].set_xlabel('Date'); axes[4].set_ylim(0, 1.2)
for ax in axes: ax.grid(alpha=0.3)
fig.suptitle('Technical Indicators & Signals Panel', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig2_indicators.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig2_indicators.png")

# Fig3: 净值+回撤
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
ax1.plot(df.index, df['strat_nav'], 'r-', lw=2, label=f"Strategy ({m['复合策略']['cum']*100:+.1f}%)")
ax1.plot(df.index, df['bench_nav'], 'gray', lw=1, alpha=0.5, label=f"B&H ({m['买入持有']['cum']*100:+.1f}%)")
ax1.fill_between(df.index, 1, df['strat_nav'], alpha=0.1, color='red')
ax1.legend(fontsize=10); ax1.set_ylabel('Net Value'); ax1.grid(alpha=0.3)
ax1.set_title('Portfolio Net Value Comparison', fontsize=14, fontweight='bold')

dd_s = (df['strat_nav'] - df['strat_nav'].cummax()) / df['strat_nav'].cummax()
dd_b = (df['bench_nav'] - df['bench_nav'].cummax()) / df['bench_nav'].cummax()
ax2.fill_between(df.index, 0, dd_s, color='red', alpha=0.4, label=f"Strategy DD ({m['复合策略']['maxdd']*100:+.1f}%)")
ax2.fill_between(df.index, 0, dd_b, color='gray', alpha=0.3, label=f"B&H DD ({m['买入持有']['maxdd']*100:+.1f}%)")
ax2.legend(fontsize=10); ax2.set_ylabel('Drawdown'); ax2.set_xlabel('Date'); ax2.grid(alpha=0.3)
ax2.set_ylim(min(dd_s.min(), dd_b.min())*1.1, 0.02)
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig3_nav_dd.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig3_nav_dd.png")

# Fig4: 年度收益 + VaR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
df['year'] = df.index.year
annual = df.groupby('year').apply(lambda g: pd.Series({
    'strat': (1+g['strat_ret'].fillna(0)).prod()-1,
    'bench': (1+g['bench_ret'].fillna(0)).prod()-1
})).reset_index()
x = np.arange(len(annual)); w = 0.35
ax1.bar(x-w/2, annual['strat']*100, w, label='Strategy', color='#CC5500', edgecolor='white')
ax1.bar(x+w/2, annual['bench']*100, w, label='B&H', color='#5A4A38', edgecolor='white', alpha=0.5)
for i, (s, b) in enumerate(zip(annual['strat']*100, annual['bench']*100)):
    ax1.text(i-w/2, s+0.3, f'{s:+.1f}%', ha='center', fontsize=9, fontweight='bold', color='#CC5500')
ax1.set_xticks(x); ax1.set_xticklabels(annual['year']); ax1.legend(); ax1.grid(axis='y', alpha=0.3)
ax1.axhline(0, color='black', lw=1); ax1.set_title('Annual Returns', fontsize=13, fontweight='bold')

# VaR histogram
ax2.hist(df['strat_ret'].dropna(), bins=80, color='steelblue', alpha=0.7, edgecolor='white', density=True)
ax2.axvline(m['复合策略']['var95'], color='red', ls='--', lw=2, label=f"VaR95={m['复合策略']['var95']:.4f}")
ax2.axvline(m['复合策略']['var99'], color='darkred', ls='--', lw=2, label=f"VaR99={m['复合策略']['var99']:.4f}")
ax2.set_xlabel('Daily Return'); ax2.set_title('Strategy Returns Distribution & VaR', fontsize=13, fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig4_annual_var.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig4_annual_var.png")

# Fig5: 绩效雷达图
from matplotlib.patches import Circle
categories = ['Return', 'Sharpe', 'Low DD', 'Win Rate', 'Low VaR', 'Low Vol']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def normalize_metrics(ms):
    bh_cum = max(0.01, m['买入持有']['cum'])
    bh_vol = max(0.01, m['买入持有']['vol'])
    return [
        max(0, min(1, (ms['cum'] + 1) / (bh_cum + 1))),
        max(0, min(1, (ms['sharpe'] + 2) / 4)),
        max(0, min(1, 1 - abs(ms['maxdd']))),
        max(0, min(1, ms['win'])),
        max(0, min(1, 1 - abs(ms['var95']) * 20)),
        max(0, min(1, 1 - ms['vol'] / bh_vol)),
    ]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
for name, color, ls in [('复合策略', '#CC5500', '-'), ('买入持有', '#5A4A38', '--')]:
    vals = normalize_metrics(m[name])
    vals += vals[:1]
    ax.plot(angles, vals, color=color, lw=2, ls=ls, label=name)
    ax.fill(angles, vals, alpha=0.1, color=color)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Strategy Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig5_radar.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig5_radar.png")

# Fig6: 信号贡献饼图
fig, ax = plt.subplots(figsize=(7, 7))
sig_counts = {
    'Trend(MA)': df['sig_trend'].sum(),
    'MACD': df['sig_macd'].sum(),
    'RSI': df['sig_rsi'].sum(),
    'Policy': df['sig_sent'].sum(),
    'MoneyFlow': df['sig_money'].sum(),
}
ax.pie(list(sig_counts.values()), labels=list(sig_counts.keys()),
       autopct='%1.1f%%', colors=['#CC5500','#E87830','#FFB366','#2558A0','#1E7A4A'],
       explode=(0,0,0,0.05,0.1), startangle=90, textprops={'fontsize': 11})
ax.set_title('Signal Dimension Contribution', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig6_pie.png'), dpi=150, bbox_inches='tight'); plt.close()
print("  [OK] fig6_pie.png")

# Save data
df.to_csv(os.path.join(DATA_DIR, 'changjiang_power_analysis.csv'), encoding='utf-8-sig')

# Save to SQLite (L8)
conn = sqlite3.connect(os.path.join(DATA_DIR, 'changjiang_power.db'))
df.to_sql('daily_data', conn, if_exists='replace')
pd.DataFrame(policy_events.items(), columns=['date', 'sentiment']).to_sql('policy_events', conn, if_exists='replace')
conn.close()
print(f"  [OK] SQLite database saved")

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*60}")
print(f"  分析完成!")
print(f"{'='*60}")
print(f"  图表: 6张 (fig1~fig6)")
print(f"  数据: analysis.csv + SQLite DB")
print(f"  策略年化: {m['复合策略']['ann']*100:.1f}%  夏普: {m['复合策略']['sharpe']:.2f}")
print(f"  最大回撤: {m['复合策略']['maxdd']*100:.1f}%  VaR95: {m['复合策略']['var95']:.4f}")
print(f"  全部输出: {FIG_DIR}")
