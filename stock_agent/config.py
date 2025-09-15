import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_agent.db')
    
    # App Settings
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Data Sources
    ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
    NEWS_API_BASE_URL = 'https://newsapi.org/v2'
    
    # Default Settings
    DEFAULT_PORTFOLIO_SIZE = 10
    DEFAULT_RISK_TOLERANCE = 'medium'
    DEFAULT_INVESTMENT_AMOUNT = 10000
    
    # Supported Sectors
    SECTORS = [
        'Technology', 'Healthcare', 'Financial', 'Consumer Discretionary',
        'Consumer Staples', 'Energy', 'Industrials', 'Materials',
        'Real Estate', 'Utilities', 'Communication Services'
    ]
    
    # Risk Levels
    RISK_LEVELS = ['low', 'medium', 'high']
    
    # Time Frames
    TIME_FRAMES = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y']
