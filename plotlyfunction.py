import plotly.graph_objects as go

def plot_k_line_chart(stock_code, df):
    try:
        # 确保包含必要字段
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("数据缺失必要字段")

        # 创建K线图
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='K线'
                )
            ]
        )
        
        # 添加信号标记
        if 'signal' in df.columns:
            buy_signals = df[df['signal'] == 'buy']
            sell_signals = df[df['signal'] == 'sell']
            
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low']*0.98,
                mode='markers',
                marker=dict(color='green', size=10),
                name='买入信号'
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High']*1.02,
                mode='markers',
                marker=dict(color='red', size=10),
                name='卖出信号'
            ))

        # 设置布局
        fig.update_layout(
            title=f'{stock_code} 股价走势与交易信号',
            xaxis_title='日期',
            yaxis_title='价格',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        return fig
        
    except Exception as e:
        print(f"绘图时发生错误: {e}")
        return go.Figure()  # 返回空图表防止程序崩溃