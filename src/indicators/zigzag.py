import pandas as pd
import numpy as np

class ZigZag:
    """
    ZigZag算法实现，用于识别价格波动的高点和低点（Swing Points）。
    这是DiNapoli分析的基础，用于确定“焦点数”（Focus Numbers）。
    """

    @staticmethod
    def calculate(high: pd.Series, low: pd.Series, deviation: float = 5.0) -> pd.Series:
        """
        计算ZigZag指标。
        
        Args:
            high (pd.Series): 最高价序列
            low (pd.Series): 最低价序列
            deviation (float): 偏差阈值（百分比），例如 5.0 代表 5%
            
        Returns:
            pd.Series: 包含ZigZag转折点的序列。非转折点为NaN，转折点为该点的价格。
                       正值代表高点，负值代表低点（为了区分，实际价格取绝对值）。
                       或者更简单：返回插值后的ZigZag线，或者仅返回关键点。
                       这里我们返回一个Series，其中关键点有值（价格），其他为NaN。
                       为了区分高低点，可以结合trend状态，但这里仅返回价格。
        """
        # 简单的ZigZag实现
        # 1. 确定初始趋势
        # 2. 遍历价格，如果反向运动超过deviation，则确认上一个极值点
        
        df = pd.DataFrame({'high': high, 'low': low})
        n = len(df)
        if n == 0:
            return pd.Series()

        # 结果序列，存储转折点
        pivots = pd.Series(np.nan, index=df.index)
        
        # 状态变量
        last_pivot_price = df['high'].iloc[0]
        last_pivot_idx = 0
        trend = 1 # 1 for Up, -1 for Down
        
        # 寻找第一个确定的趋势方向（可选，这里简化处理，假设从第一个点开始）
        
        for i in range(1, n):
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            
            if trend == 1: # 当前是上升趋势，寻找更高的高点
                if curr_high > last_pivot_price:
                    last_pivot_price = curr_high
                    last_pivot_idx = i
                elif curr_low < last_pivot_price * (1 - deviation / 100):
                    # 反转：价格下跌超过阈值，确认之前的高点
                    pivots.iloc[last_pivot_idx] = last_pivot_price
                    trend = -1
                    last_pivot_price = curr_low
                    last_pivot_idx = i
            else: # 当前是下降趋势，寻找更低的低点
                if curr_low < last_pivot_price:
                    last_pivot_price = curr_low
                    last_pivot_idx = i
                elif curr_high > last_pivot_price * (1 + deviation / 100):
                    # 反转：价格上涨超过阈值，确认之前的低点
                    pivots.iloc[last_pivot_idx] = last_pivot_price
                    trend = 1
                    last_pivot_price = curr_high
                    last_pivot_idx = i
                    
        # 最后一个点通常作为临时极值
        pivots.iloc[last_pivot_idx] = last_pivot_price
        
        return pivots

    @staticmethod
    def get_swing_points(high: pd.Series, low: pd.Series, deviation: float = 5.0) -> list:
        """
        获取结构化的Swing Points列表。
        
        Returns:
            list of dict: [{'date': timestamp, 'price': float, 'type': 'high'/'low'}]
        """
        pivots = ZigZag.calculate(high, low, deviation)
        swing_points = []
        
        # 需要重新遍历来确定是高点还是低点
        # 简单方法：如果该点是局部最大值则是高点，局部最小值则是低点
        # 但由于ZigZag性质，高低点是交替的
        
        valid_indices = pivots.dropna().index
        if len(valid_indices) < 2:
            return []
            
        # 判断第一个点类型
        first_idx = valid_indices[0]
        second_idx = valid_indices[1]
        
        # 如果第一个点价格 < 第二个点价格，则第一个点是低点
        if pivots[first_idx] < pivots[second_idx]:
            current_type = 'low'
        else:
            current_type = 'high'
            
        for idx in valid_indices:
            swing_points.append({
                'date': idx,
                'price': pivots[idx],
                'type': current_type
            })
            # 切换类型
            current_type = 'low' if current_type == 'high' else 'high'
            
        return swing_points
