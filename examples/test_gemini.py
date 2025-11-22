import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.llm_analyst import LLMAnalyst

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing Gemini API Connection...")
    
    analyst = LLMAnalyst()
    
    context = {
        'symbol': 'TEST',
        'trend': 'Up',
        'volatility': 'Low',
        'pattern': 'None',
        'rsi': '50',
        'close': '100'
    }
    
    print("Sending request...")
    result = analyst.analyze_context(context)
    print("\nResult:")
    print(result)

if __name__ == "__main__":
    main()
