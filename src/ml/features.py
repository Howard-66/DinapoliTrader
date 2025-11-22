import pandas as pd
import numpy as np
from src.indicators.basics import Indicators

class FeatureEngineer:
    """
    特征工程模块，用于为机器学习模型准备数据。
    生成DiNapoli相关特征以及通用技术指标特征。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_features(self) -> pd.DataFrame:
        """
        生成特征集。
        
        Returns:
            pd.DataFrame: 包含原始数据和新特征的DataFrame。
        """
        df = self.df
        
        # 1. 基础指标
        df['rsi'] = Indicators.rsi(df['close'], 14)
        df['sma_50'] = Indicators.sma(df['close'], 50)
        df['sma_200'] = Indicators.sma(df['close'], 200)
        
        # 2. DiNapoli 特征
        # 距离DMA的距离（百分比）
        dma_3x3 = Indicators.displaced_ma(df['close'], 3, 3)
        dma_25x5 = Indicators.displaced_ma(df['close'], 25, 5)
        
        df['dist_dma3'] = (df['close'] - dma_3x3) / dma_3x3
        df['dist_dma25'] = (df['close'] - dma_25x5) / dma_25x5
        
        # 3. 波动率特征
        # 简单波动率 (Close / Open - 1)
        df['volatility'] = (df['high'] - df['low']) / df['open']
        
        # 4. 趋势特征
        # 25x5 斜率 (简单用当前值 - 5天前值)
        df['trend_slope'] = dma_25x5.diff(5)
        
        # 5. 目标变量 (Label)
        # 预测未来N天的收益率是否 > 阈值
        # 这里仅生成特征，Label生成通常在训练阶段做，或者这里也可以做
        
        return df.dropna()

    def create_labels(self, horizon: int = 5, threshold: float = 0.02) -> pd.Series:
        """
        创建分类标签。
        1: 未来horizon天内最大收益 > threshold
        0: 否则
        """
        # 这是一个简化的Label逻辑，实际可能需要考虑止损
        # Future Return = (Future Close - Current Close) / Current Close
        
        future_returns = self.df['close'].shift(-horizon) / self.df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        return labels
