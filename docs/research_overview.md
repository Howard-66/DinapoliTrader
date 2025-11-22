# Research Overview: DiNapoli Levels & Quantitative Adaptation

## 1. The Subjectivity Problem
Traditional DiNapoli analysis relies heavily on the trader's "eye" to pick the correct Swing Highs/Lows for Fibonacci drawings.
**Solution**: Use a **ZigZag algorithm** with dynamic depth (or fractal-based swing detection) to objectively identify swing points.

## 2. Core Components

### A. Displaced Moving Averages (DMA)
- **3x3 DMA**: 3-period SMA shifted forward by 3 periods. Acts as immediate support/resistance and trend filter.
- **7x5 DMA**: 7-period SMA shifted forward by 5 periods. Intermediate trend.
- **25x5 DMA**: 25-period SMA shifted forward by 5 periods. Major trend direction.

### B. Patterns
- **Double Repo (Double Repenetration)**: A reversal pattern.
    - *Quant Logic*: Price closes below 3x3, makes a low, closes back above, makes a higher low (or lower low but fails to sustain), closes back above.
    - *Confirmation*: Must happen "far" from the 25x5 DMA (overextended).
- **Single Penetration**: Trend continuation.
    - *Quant Logic*: Strong thrust (price > 3x3 for N bars), then pulls back to 0.382 or 0.618 level, without closing below the 3x3 for more than M bars.

## 3. Machine Learning Integration
Instead of hard-coded rules for "overextended", we use ML:
- **Features**: Distance(Price, 25x5), Slope(25x5), Volatility(ATR), RSI.
- **Label**: 1 if Trade hits Target before Stop, 0 otherwise.
- **Model**: Gradient Boosting (XGBoost/LightGBM).

## 4. LLM Role
LLMs are not good at precise math, but good at **Context Synthesis**.
- **Input**: "Trend is Up on Daily, Down on Hourly. Double Repo detected on Hourly. Major economic news in 1 hour."
- **Output**: "Risk Level: High. Recommendation: Skip or Reduce Size."
