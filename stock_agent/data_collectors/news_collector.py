import requests
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
from config import Config

class NewsCollector:
    """뉴스 데이터 수집 클래스"""
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key or Config.NEWS_API_KEY
        self.base_url = Config.NEWS_API_BASE_URL
        
    def get_market_news(self, query: str = 'stock market', 
                       language: str = 'en', 
                       sort_by: str = 'publishedAt',
                       page_size: int = 20) -> List[Dict]:
        """시장 뉴스 조회"""
        if not self.news_api_key:
            return self._get_rss_news(query)
            
        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}/everything", params=params)
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'content': article.get('content', '')
                    })
                return articles
        except Exception as e:
            print(f"News API error: {e}")
            return self._get_rss_news(query)
        
        return []
    
    def get_stock_news(self, symbol: str) -> List[Dict]:
        """특정 주식 관련 뉴스 조회"""
        query = f"{symbol} stock"
        return self.get_market_news(query)
    
    def get_sector_news(self, sector: str) -> List[Dict]:
        """특정 섹터 관련 뉴스 조회"""
        query = f"{sector} sector stocks"
        return self.get_market_news(query)
    
    def get_economic_news(self) -> List[Dict]:
        """경제 뉴스 조회"""
        query = "federal reserve economy inflation GDP unemployment"
        return self.get_market_news(query)
    
    def _get_rss_news(self, query: str) -> List[Dict]:
        """RSS 피드를 통한 뉴스 수집 (API 키가 없을 때)"""
        rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.marketwatch.com/marketwatch/topstories/',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.reuters.com/news/wealth',
            'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
        ]
        
        articles = []
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:  # 각 피드에서 최대 5개 기사
                    if query.lower() in entry.title.lower() or query.lower() in entry.get('summary', '').lower():
                        articles.append({
                            'title': entry.title,
                            'description': entry.get('summary', ''),
                            'url': entry.link,
                            'published_at': entry.get('published', ''),
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'content': ''
                        })
            except Exception as e:
                print(f"RSS feed error for {feed_url}: {e}")
                continue
                
        return articles[:20]  # 최대 20개 기사 반환
    
    def get_sentiment_analysis(self, text: str) -> Dict:
        """텍스트 감정 분석 (간단한 키워드 기반)"""
        positive_keywords = [
            'bullish', 'growth', 'profit', 'gain', 'rise', 'increase', 
            'positive', 'strong', 'outperform', 'beat', 'exceed'
        ]
        negative_keywords = [
            'bearish', 'decline', 'loss', 'fall', 'drop', 'decrease',
            'negative', 'weak', 'underperform', 'miss', 'disappoint'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = 'neutral'
            score = 0.5
            
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }
    
    def get_news_sentiment(self, symbol: str) -> Dict:
        """주식 관련 뉴스의 감정 분석"""
        news_articles = self.get_stock_news(symbol)
        
        if not news_articles:
            return {'sentiment': 'neutral', 'score': 0.5, 'article_count': 0}
        
        sentiments = []
        for article in news_articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.get_sentiment_analysis(text)
            sentiments.append(sentiment)
        
        # 평균 감정 점수 계산
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        
        if positive_count > negative_count:
            overall_sentiment = 'positive'
        elif negative_count > positive_count:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'sentiment': overall_sentiment,
            'score': avg_score,
            'article_count': len(news_articles),
            'positive_articles': positive_count,
            'negative_articles': negative_count
        }
