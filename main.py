
import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
from data import read_day_from_tushare, select_time
from backtest import backtest_results
from plotlyfunction import plot_k_line_chart
from openai import OpenAI
import time
import plotly.graph_objects as go
# 设置页面配置
st.set_page_config(
    page_title="股票策略回测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置 API Token
def set_tushare_token():
    ts.set_token('c5c5700a6f4678a1837ad234f2e9ea2a573a26b914b47fa2dbb38aff')
    return ts.pro_api()

# 获取AI生成的代码
def get_generated_code(prompt_buy, prompt_sell, retries=3, delay=5):
    prompt = f"""
    生成一个py函数，函数名为generate_signal()，输入参数为df类型的股票日度行情数据，索引为日期，列名为Open, High, Low, Close, Volume, Amount。
    根据下面的交易逻辑生成df['signal']列：
    买入条件：当{prompt_buy}时，df['signal']='buy'；
    卖出条件：当{prompt_sell}线时，df['signal'] = 'sell'；
    请注意以下几点：
    1. 请使用 `.at[]` 或 `.loc[]` 来逐行赋值 `df['signal']`，避免使用 `.iloc[]` 进行赋值；
    2. 默认情况下，`df['signal']` 列应为 `None`；
    3. 确保函数返回修改后的 `df`，并且 `df['signal']` 列只包含 `'buy'` 或 `'sell'`。
    请仅生成代码，不需要解释。我需要再程序中使用exec动态执行你的回答"""

    try:
        client = OpenAI(api_key="sk-1e63e70de8e5442594186ee9cf8e9ee6", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model="qwen-max-latest",
                    messages=[
                        {'role': 'system', 'content': '你是专业的证券分析师和老练的程序员'},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    print(f"请求失败，重试 {attempt + 1}/{retries}...")
                    time.sleep(delay)  # 延时重试
                else:
                    print(f"错误信息：{e}")
                    return None
    except Exception as e:
        print(f"请求失败：{e}")
        return None

# 执行生成的代码
def execute_generated_code(generated_code):
    try:
        # 清理代码中的多余标记（例如 ```python）
        generated_code = generated_code.strip("```python").strip("```").strip()

        # 输出清理后的代码，查看AI返回的代码
        print("生成的代码：")
        print(generated_code)

        # 使用exec动态执行生成的代码，并明确作用域
        exec(generated_code, globals(), locals())

        # 验证生成的函数是否存在
        if 'generate_signal' in locals():
            return locals()['generate_signal']
        else:
            print("生成的代码失败，请重试。")
            return None
    except Exception as e:
        print(f"执行代码时发生错误: {e}")
        return None

# 获取股票数据并处理（添加缓存）
@st.cache_data
def get_stock_data(stock_code, start_date, end_date):
    df = read_day_from_tushare(stock_code, symbol_type='stock')
    df['signal'] = np.nan
    df = select_time(df, start_time=start_date, end_time=end_date)
    return df

# 显示回测结果
def display_backtest_results(result):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("同期标的涨跌幅", f"{result['同期标的涨跌幅']*100:.2f}%")
        st.metric("累计收益率", f"{result['"波段盈"累计收益率']*100:.2f}%")
    with col2:
        st.metric("超额收益率", f"{result['超额收益率']*100:.2f}%")
        st.metric("最大回撤", f"{result['最大回撤']*100:.2f}%")
    with col3:
        st.metric("胜率", f"{result['胜率']*100:.2f}%")
        st.metric("交易笔数", result['交易笔数'])
    with col4:
        st.metric("平均收益率", f"{result['单笔平均收益率']*100:.2f}%")
        st.metric("持仓天数", result['持仓天数'])

# 主程序
def main():
    st.title("📈 智能股票策略回测系统")
    
    # 侧边栏输入
    with st.sidebar:
        st.header("参数设置")
        stock_code = st.text_input("股票代码", "601555.SH")
        start_date = st.text_input("开始日期", "20230101")
        end_date = st.text_input("结束日期", "20240910")
        prompt_buy = st.text_input("买入条件", "5日线上穿10日线")
        prompt_sell = st.text_input("卖出条件", "10日线上穿5日")
        run_btn = st.button("运行回测")
    
    if run_btn:
        with st.spinner("正在获取数据并生成策略..."):
            try:
                # 获取数据
                df = get_stock_data(stock_code, start_date, end_date)
                
                # 生成策略代码
                generated_code = get_generated_code(prompt_buy, prompt_sell)
                
                if generated_code:
                    generate_signal = execute_generated_code(generated_code)
                    
                    if generate_signal:
                        df = generate_signal(df)
                        
                        # 执行回测
                        signal_df = df[df['signal'].isin(['buy', 'sell'])]
                        result = backtest_results(df=df, signal_df=signal_df, initial_capital=1_000_000)
                        
                        # 显示结果
                        st.subheader("回测结果")
                        display_backtest_results(result)
                        
                        # 显示详细数据
                        with st.expander("查看详细交易记录"):
                            st.dataframe(signal_df[['Open', 'High', 'Low', 'Close', 'signal']])
                        
                        # 显示K线图
                        if not df.empty:
                            try:
                                st.subheader("K线图与交易信号")
                                fig = plot_k_line_chart(stock_code, df)
                                
                                # 添加类型验证
                                if isinstance(fig, go.Figure) and len(fig.data) > 0:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("无法生成有效的K线图表")
                                    
                            except Exception as e:
                                st.error(f"图表生成失败: {str(e)}")
                    else:
                        st.error("策略生成失败，请检查输入条件")
                else:
                    st.error("未能生成有效策略代码，请重试")
                    
            except Exception as e:
                st.error(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main()
