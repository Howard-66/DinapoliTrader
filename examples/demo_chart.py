import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.feed import DataFeed
from src.indicators.basics import Indicators
from src.utils.visualization import Visualizer

def main():
    # 1. Fetch Data (using yfinance if available, else mock)
    feed = DataFeed()
    try:
        df = feed.fetch_data('BTC-USD', '2023-01-01', '2023-06-01')
        if df.empty:
            raise Exception("No data fetched")
    except Exception as e:
        print(f"Could not fetch real data: {e}. Using synthetic data.")
        dates = pd.date_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    # 2. Calculate Indicators
    # DiNapoli 3x3 DMA
    dma_3x3 = Indicators.displaced_ma(df['close'], period=3, displacement=3)
    # DiNapoli 7x5 DMA
    dma_7x5 = Indicators.displaced_ma(df['close'], period=7, displacement=5)
    # DiNapoli 25x5 DMA
    dma_25x5 = Indicators.displaced_ma(df['close'], period=25, displacement=5)
    
    indicators = {
        'DMA 3x3': dma_3x3,
        'DMA 7x5': dma_7x5,
        'DMA 25x5': dma_25x5
    }

    # 3. Visualize
    fig = Visualizer.plot_chart(df, indicators, title="BTC-USD DiNapoli Chart")
    
    output_file = 'examples/btc_dinapoli.html'
    fig.write_html(output_file)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    main()
