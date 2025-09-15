from typing import List, Dict, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os

class NotificationService:
    """알림 서비스"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = 587, 
                 email: str = None, password: str = None):
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.email = email or os.getenv('NOTIFICATION_EMAIL')
        self.password = password or os.getenv('NOTIFICATION_PASSWORD')
        self.notifications_enabled = bool(self.email and self.password)
    
    def send_price_alert(self, symbol: str, current_price: float, 
                        target_price: float, alert_type: str = 'above') -> bool:
        """가격 알림 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Price Alert: {symbol}"
            
            if alert_type == 'above':
                message = f"{symbol} is now trading at ${current_price:.2f}, above your target of ${target_price:.2f}"
            else:
                message = f"{symbol} is now trading at ${current_price:.2f}, below your target of ${target_price:.2f}"
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending price alert: {e}")
            return False
    
    def send_risk_alert(self, portfolio_id: int, risk_level: str, 
                       risk_metrics: Dict) -> bool:
        """리스크 알림 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Risk Alert: Portfolio {portfolio_id}"
            
            message = f"""
            Portfolio {portfolio_id} Risk Alert
            
            Risk Level: {risk_level}
            
            Key Metrics:
            - Volatility: {risk_metrics.get('volatility', 0):.2%}
            - VaR (95%): {risk_metrics.get('var_95', 0):.2%}
            - Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}
            - Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}
            
            Please review your portfolio allocation.
            """
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending risk alert: {e}")
            return False
    
    def send_rebalance_alert(self, portfolio_id: int, rebalance_info: Dict) -> bool:
        """리밸런싱 알림 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Rebalancing Alert: Portfolio {portfolio_id}"
            
            message = f"""
            Portfolio {portfolio_id} Rebalancing Alert
            
            Rebalancing is recommended based on current market conditions.
            
            Suggested Changes:
            """
            
            for symbol, change in rebalance_info.get('weight_changes', {}).items():
                if abs(change) > 0.01:  # 1% 이상 변화
                    message += f"- {symbol}: {change:+.1%}\n"
            
            message += f"\nTotal Turnover: {rebalance_info.get('total_turnover', 0):.1%}"
            message += f"\nEstimated Cost: ${rebalance_info.get('estimated_cost', 0):.2f}"
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending rebalance alert: {e}")
            return False
    
    def send_news_alert(self, symbol: str, news_items: List[Dict]) -> bool:
        """뉴스 알림 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"News Alert: {symbol}"
            
            message = f"Recent news for {symbol}:\n\n"
            
            for news in news_items[:5]:  # 최대 5개 기사
                message += f"• {news.get('title', 'No title')}\n"
                message += f"  Source: {news.get('source', 'Unknown')}\n"
                message += f"  Sentiment: {news.get('sentiment', 'Neutral')}\n\n"
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending news alert: {e}")
            return False
    
    def send_performance_report(self, portfolio_id: int, performance_data: Dict) -> bool:
        """성과 리포트 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Performance Report: Portfolio {portfolio_id}"
            
            message = f"""
            Portfolio {portfolio_id} Performance Report
            
            Period: {performance_data.get('period', 'N/A')}
            
            Performance Metrics:
            - Total Return: {performance_data.get('total_return', 0):.2%}
            - Volatility: {performance_data.get('volatility', 0):.2%}
            - Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}
            - Max Drawdown: {performance_data.get('max_drawdown', 0):.2%}
            
            Top Performers:
            """
            
            for performer in performance_data.get('top_performers', [])[:3]:
                message += f"- {performer.get('symbol', 'N/A')}: {performer.get('return', 0):.2%}\n"
            
            message += "\nUnderperformers:\n"
            
            for underperformer in performance_data.get('underperformers', [])[:3]:
                message += f"- {underperformer.get('symbol', 'N/A')}: {underperformer.get('return', 0):.2%}\n"
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending performance report: {e}")
            return False
    
    def send_market_alert(self, market_condition: str, market_data: Dict) -> bool:
        """시장 알림 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Market Alert: {market_condition}"
            
            message = f"""
            Market Condition Alert
            
            Current Market State: {market_condition}
            
            Key Indicators:
            - S&P 500: {market_data.get('sp500_change', 0):.2%}
            - VIX: {market_data.get('vix_level', 0):.2f}
            - Market Sentiment: {market_data.get('sentiment', 'Neutral')}
            
            Recommendations:
            {market_data.get('recommendations', 'Monitor your positions closely.')}
            """
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending market alert: {e}")
            return False
    
    def send_daily_summary(self, portfolio_id: int, summary_data: Dict) -> bool:
        """일일 요약 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            subject = f"Daily Summary: Portfolio {portfolio_id}"
            
            message = f"""
            Daily Portfolio Summary - {datetime.now().strftime('%Y-%m-%d')}
            
            Portfolio Value: ${summary_data.get('total_value', 0):,.2f}
            Daily Change: {summary_data.get('daily_change', 0):.2%}
            Total Return: {summary_data.get('total_return', 0):.2%}
            
            Top Movers:
            """
            
            for mover in summary_data.get('top_movers', [])[:5]:
                message += f"- {mover.get('symbol', 'N/A')}: {mover.get('change', 0):.2%}\n"
            
            message += f"\nMarket Overview:\n"
            message += f"- S&P 500: {summary_data.get('market_change', 0):.2%}\n"
            message += f"- VIX: {summary_data.get('vix', 0):.2f}\n"
            
            return self._send_email(subject, message)
            
        except Exception as e:
            print(f"Error sending daily summary: {e}")
            return False
    
    def _send_email(self, subject: str, message: str) -> bool:
        """이메일 전송"""
        try:
            if not self.notifications_enabled:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email  # 실제 구현에서는 사용자 이메일 사용
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            
            text = msg.as_string()
            server.sendmail(self.email, self.email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def create_alert_rule(self, rule_type: str, conditions: Dict, 
                         user_email: str = None) -> bool:
        """알림 규칙 생성"""
        try:
            # 실제 구현에서는 데이터베이스에 저장
            alert_rule = {
                'id': f"rule_{datetime.now().timestamp()}",
                'type': rule_type,
                'conditions': conditions,
                'user_email': user_email or self.email,
                'created_at': datetime.now().isoformat(),
                'active': True
            }
            
            # 규칙 저장 (실제 구현에서는 데이터베이스 사용)
            print(f"Alert rule created: {alert_rule}")
            return True
            
        except Exception as e:
            print(f"Error creating alert rule: {e}")
            return False
    
    def check_alert_conditions(self, portfolio_id: int, market_data: Dict) -> List[Dict]:
        """알림 조건 확인"""
        try:
            triggered_alerts = []
            
            # 가격 알림 확인
            for symbol, data in market_data.get('stock_prices', {}).items():
                current_price = data.get('price', 0)
                
                # 실제 구현에서는 사용자 설정된 알림 규칙 확인
                # 예시: 5% 이상 상승/하락
                if abs(data.get('change_percent', 0)) > 5:
                    triggered_alerts.append({
                        'type': 'price_alert',
                        'symbol': symbol,
                        'message': f"{symbol} moved {data.get('change_percent', 0):.2%}",
                        'priority': 'medium'
                    })
            
            # 리스크 알림 확인
            portfolio_risk = market_data.get('portfolio_risk', {})
            if portfolio_risk.get('volatility', 0) > 0.3:
                triggered_alerts.append({
                    'type': 'risk_alert',
                    'portfolio_id': portfolio_id,
                    'message': f"High volatility detected: {portfolio_risk.get('volatility', 0):.2%}",
                    'priority': 'high'
                })
            
            # 시장 알림 확인
            market_condition = market_data.get('market_condition', 'normal')
            if market_condition in ['high_volatility', 'market_stress']:
                triggered_alerts.append({
                    'type': 'market_alert',
                    'message': f"Market condition: {market_condition}",
                    'priority': 'high'
                })
            
            return triggered_alerts
            
        except Exception as e:
            print(f"Error checking alert conditions: {e}")
            return []
    
    def send_bulk_notifications(self, notifications: List[Dict]) -> Dict:
        """대량 알림 전송"""
        try:
            results = {
                'total': len(notifications),
                'successful': 0,
                'failed': 0,
                'errors': []
            }
            
            for notification in notifications:
                try:
                    if notification['type'] == 'price_alert':
                        success = self.send_price_alert(
                            notification['symbol'],
                            notification['current_price'],
                            notification['target_price'],
                            notification.get('alert_type', 'above')
                        )
                    elif notification['type'] == 'risk_alert':
                        success = self.send_risk_alert(
                            notification['portfolio_id'],
                            notification['risk_level'],
                            notification['risk_metrics']
                        )
                    elif notification['type'] == 'news_alert':
                        success = self.send_news_alert(
                            notification['symbol'],
                            notification['news_items']
                        )
                    else:
                        success = False
                        results['errors'].append(f"Unknown notification type: {notification['type']}")
                    
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error sending {notification['type']}: {str(e)}")
            
            return results
            
        except Exception as e:
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'errors': [f"Bulk notification error: {str(e)}"]
            }
