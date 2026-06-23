# changjiang_power_factor_strategy
长江电力(600900)多因子量化择时课程设计项目
## 项目文件说明
- analysis.py：多因子策略核心计算，股价指标、政策情感、资金流向复合信号回测
- dashboard.py：Streamlit交互式可视化看板，展示行情、因子曲线、回测结果
- requirements.txt：项目Python依赖清单
- changjiang_power_analysis.csv：长江电力历史交易日行情数据
- changjiang_power.db：本地SQLite行情数据库

## 本地运行步骤
1. 安装依赖
```bash
pip install -r requirements.txt

2. 启动可视化看板
```bash
streamlit run dashboard.py

## 项目研究背景
本课程设计以长江电力(600900)水电龙头标的为研究对象，构建三类复合量化择时因子：
1. 技术行情因子：股价均线、波动率、日收益率等技术指标
2. 行业舆情因子：水电政策新闻情感量化打分
3. 资金流向因子：个股主力资金净流入、筹码变动信号
依托Streamlit搭建交互式网页可视化平台，直观呈现历史行情、因子走势与策略回测收益曲线，用于课程设计答辩演示。
