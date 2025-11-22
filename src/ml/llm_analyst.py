import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LLMAnalyst:
    """
    LLM分析师 (Gemini Powered)。
    负责分析非结构化数据（新闻、宏观环境）或提供“第二意见”。
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. LLM Analyst will be disabled or mock.")
        else:
            logger.info(f"Gemini API Key found: {self.api_key[:4]}***")
            genai.configure(api_key=self.api_key)
            # Use gemini-flash-latest as it is listed in available models
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_context(self, market_context: dict) -> str:
        """
        分析市场环境。
        
        Args:
            market_context (dict): 包含趋势、波动率、新闻摘要等信息的字典。
            
        Returns:
            str: LLM的分析建议。
        """
        if not hasattr(self, 'model'):
             return "LLM Analyst not configured (Missing API Key)."

        prompt = f"""
        You are a professional quantitative trader specializing in Joe DiNapoli's Fibonacci analysis.
        
        Analyze the following market context for {market_context.get('symbol', 'Unknown Asset')}:
        
        - **Trend (25x5 DMA)**: {market_context.get('trend', 'Unknown')}
        - **Volatility**: {market_context.get('volatility', 'Unknown')}
        - **Recent Pattern**: {market_context.get('pattern', 'None')}
        - **RSI (14)**: {market_context.get('rsi', 'Unknown')}
        - **Close Price**: {market_context.get('close', 'Unknown')}
        
        Provide a concise trading recommendation (Buy/Sell/Hold) and risk assessment. 
        Focus on whether the DiNapoli pattern (if any) is supported by the context.
        Limit response to 3 sentences.
        """
        
        try:
            logger.info("Sending prompt to Gemini...")
            response = self.model.generate_content(prompt)
            logger.info(f"Gemini Response received: {response.text[:100]}...")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return f"Error analyzing context: {e}"

    def get_risk_adjustment(self, market_context: dict) -> float:
        """
        获取风险调整系数 (0.0 - 1.0).
        """
        # Mock logic
        if market_context.get('volatility') == 'High':
            return 0.5 # Reduce size by half
        return 1.0
