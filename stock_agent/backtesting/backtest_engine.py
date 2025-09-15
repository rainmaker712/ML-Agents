import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, initial_capital: float = 100000, 
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.0005):  # 0.05%
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """백테스트 상태 초기화"""
        self.capital = self.initial_capital
        self.positions = {}  # {symbol: shares}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.current_date = None
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    strategy: Callable, 
                    start_date: str = None,
                    end_date: str = None) -> Dict:
        """백테스트 실행"""
        self.reset()
        
        if not data:
            return {'success': False, 'message': 'No data provided'}
        
        # 공통 날짜 범위 찾기
        common_dates = self._get_common_dates(data, start_date, end_date)
        if len(common_dates) == 0:
            return {'success': False, 'message': 'No common dates found'}
        
        # 백테스트 실행
        for date in common_dates:
            self.current_date = date
            
            # 현재 가격 데이터
            current_prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'Close']
            
            if not current_prices:
                continue
            
            # 포트폴리오 가치 계산
            self._update_portfolio_value(current_prices)
            
            # 전략 실행
            try:
                signals = strategy(date, current_prices, self.positions, self.portfolio_value)
                if signals:
                    self._execute_signals(signals, current_prices)
            except Exception as e:
                print(f"Strategy error on {date}: {e}")
                continue
            
            # 포트폴리오 히스토리 기록
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy(),
                'daily_return': self._calculate_daily_return()
            })
        
        # 결과 분석
        results = self._analyze_results(data)
        return results
    
    def _get_common_dates(self, data: Dict[str, pd.DataFrame], 
                         start_date: str = None, end_date: str = None) -> List:
        """공통 날짜 범위 찾기"""
        all_dates = set()
        
        for symbol, df in data.items():
            if not df.empty:
                all_dates.update(df.index)
        
        if not all_dates:
            return []
        
        # 날짜 정렬
        sorted_dates = sorted(list(all_dates))
        
        # 시작/종료 날짜 필터링
        if start_date:
            start_dt = pd.to_datetime(start_date)
            sorted_dates = [d for d in sorted_dates if d >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            sorted_dates = [d for d in sorted_dates if d <= end_dt]
        
        return sorted_dates
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """포트폴리오 가치 업데이트"""
        position_value = 0
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                position_value += shares * current_prices[symbol]
        
        self.portfolio_value = self.cash + position_value
    
    def _execute_signals(self, signals: Dict[str, float], current_prices: Dict[str, float]):
        """거래 신호 실행"""
        for symbol, target_weight in signals.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            current_shares = self.positions.get(symbol, 0)
            current_value = current_shares * current_price
            current_weight = current_value / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # 목표 가치 계산
            target_value = self.portfolio_value * target_weight
            target_shares = target_value / current_price
            
            # 거래량 계산
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) < 0.01:  # 최소 거래량
                continue
            
            # 거래 실행
            self._execute_trade(symbol, shares_to_trade, current_price)
    
    def _execute_trade(self, symbol: str, shares: float, price: float):
        """개별 거래 실행"""
        if shares == 0:
            return
        
        # 슬리피지 적용
        if shares > 0:  # 매수
            execution_price = price * (1 + self.slippage)
        else:  # 매도
            execution_price = price * (1 - self.slippage)
        
        # 거래 금액
        trade_value = abs(shares) * execution_price
        
        # 수수료
        commission_cost = trade_value * self.commission
        
        # 총 비용
        total_cost = trade_value + commission_cost
        
        if shares > 0:  # 매수
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
                
                self.trades.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission_cost
                })
        else:  # 매도
            current_shares = self.positions.get(symbol, 0)
            if current_shares >= abs(shares):
                self.cash += trade_value - commission_cost
                self.positions[symbol] = current_shares + shares
                
                if self.positions[symbol] <= 0:
                    del self.positions[symbol]
                
                self.trades.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': abs(shares),
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission_cost
                })
    
    def _calculate_daily_return(self) -> float:
        """일일 수익률 계산"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        prev_value = self.portfolio_history[-2]['portfolio_value']
        current_value = self.portfolio_value
        
        if prev_value > 0:
            return (current_value - prev_value) / prev_value
        return 0.0
    
    def _analyze_results(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """백테스트 결과 분석"""
        if not self.portfolio_history:
            return {'success': False, 'message': 'No portfolio history'}
        
        # 포트폴리오 가치 시계열
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        dates = [h['date'] for h in self.portfolio_history]
        
        # 수익률 계산
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # 기본 통계
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 거래 통계
        trade_stats = self._analyze_trades()
        
        # 벤치마크 비교 (S&P 500)
        benchmark_return = self._calculate_benchmark_return(dates, data)
        
        return {
            'success': True,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'trade_stats': trade_stats,
            'benchmark_return': benchmark_return,
            'excess_return': annualized_return - benchmark_return,
            'portfolio_history': self.portfolio_history,
            'trades': self.trades
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """최대 낙폭 계산"""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _analyze_trades(self) -> Dict:
        """거래 통계 분석"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'total_commission': 0.0
            }
        
        total_trades = len(self.trades)
        total_commission = sum(trade['commission'] for trade in self.trades)
        
        # 거래 수익률 계산 (간단한 버전)
        trade_returns = []
        for trade in self.trades:
            if trade['action'] == 'SELL':
                # 매도 거래의 경우 수익률 계산 (실제 구현에서는 더 정교하게)
                trade_returns.append(0.0)  # 임시
        
        winning_trades = len([r for r in trade_returns if r > 0])
        losing_trades = len([r for r in trade_returns if r < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_commission': total_commission
        }
    
    def _calculate_benchmark_return(self, dates: List, data: Dict[str, pd.DataFrame]) -> float:
        """벤치마크 수익률 계산"""
        # S&P 500 데이터 찾기
        sp500_data = None
        for symbol, df in data.items():
            if '^GSPC' in symbol or 'SPY' in symbol:
                sp500_data = df
                break
        
        if sp500_data is None or sp500_data.empty:
            return 0.0
        
        # 공통 날짜의 S&P 500 수익률 계산
        sp500_values = []
        for date in dates:
            if date in sp500_data.index:
                sp500_values.append(sp500_data.loc[date, 'Close'])
        
        if len(sp500_values) < 2:
            return 0.0
        
        sp500_return = (sp500_values[-1] - sp500_values[0]) / sp500_values[0]
        annualized_sp500_return = (1 + sp500_return) ** (252 / len(sp500_values)) - 1
        
        return annualized_sp500_return
    
    def run_walk_forward_analysis(self, data: Dict[str, pd.DataFrame],
                                 strategy: Callable,
                                 train_period: int = 252,  # 1년
                                 test_period: int = 63,    # 3개월
                                 step_size: int = 21) -> Dict:  # 1개월
        """워크 포워드 분석"""
        results = []
        
        # 전체 기간을 훈련/테스트 구간으로 분할
        all_dates = self._get_common_dates(data)
        if len(all_dates) < train_period + test_period:
            return {'success': False, 'message': 'Insufficient data for walk-forward analysis'}
        
        start_idx = 0
        while start_idx + train_period + test_period <= len(all_dates):
            # 훈련 기간
            train_dates = all_dates[start_idx:start_idx + train_period]
            train_data = self._filter_data_by_dates(data, train_dates)
            
            # 테스트 기간
            test_dates = all_dates[start_idx + train_period:start_idx + train_period + test_period]
            test_data = self._filter_data_by_dates(data, test_dates)
            
            # 테스트 실행
            test_result = self.run_backtest(test_data, strategy)
            
            if test_result['success']:
                results.append({
                    'train_start': train_dates[0],
                    'train_end': train_dates[-1],
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'test_return': test_result['total_return'],
                    'test_sharpe': test_result['sharpe_ratio'],
                    'test_max_dd': test_result['max_drawdown']
                })
            
            start_idx += step_size
        
        # 결과 분석
        if not results:
            return {'success': False, 'message': 'No valid test periods'}
        
        test_returns = [r['test_return'] for r in results]
        test_sharpes = [r['test_sharpe'] for r in results]
        test_max_dds = [r['test_max_dd'] for r in results]
        
        return {
            'success': True,
            'num_periods': len(results),
            'avg_return': np.mean(test_returns),
            'avg_sharpe': np.mean(test_sharpes),
            'avg_max_dd': np.mean(test_max_dds),
            'return_std': np.std(test_returns),
            'sharpe_std': np.std(test_sharpes),
            'periods': results
        }
    
    def _filter_data_by_dates(self, data: Dict[str, pd.DataFrame], 
                             dates: List) -> Dict[str, pd.DataFrame]:
        """날짜별 데이터 필터링"""
        filtered_data = {}
        for symbol, df in data.items():
            filtered_df = df[df.index.isin(dates)]
            if not filtered_df.empty:
                filtered_data[symbol] = filtered_df
        return filtered_data
    
    def run_monte_carlo_simulation(self, data: Dict[str, pd.DataFrame],
                                  strategy: Callable,
                                  num_simulations: int = 1000,
                                  confidence_levels: List[float] = [0.05, 0.25, 0.75, 0.95]) -> Dict:
        """몬테카를로 시뮬레이션"""
        if not data:
            return {'success': False, 'message': 'No data provided'}
        
        # 수익률 데이터 추출
        returns_data = {}
        for symbol, df in data.items():
            if not df.empty and 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            return {'success': False, 'message': 'No returns data available'}
        
        # 시뮬레이션 실행
        simulation_results = []
        
        for _ in range(num_simulations):
            # 랜덤 시나리오 생성
            scenario_data = self._generate_random_scenario(returns_data)
            
            # 백테스트 실행
            result = self.run_backtest(scenario_data, strategy)
            
            if result['success']:
                simulation_results.append(result['total_return'])
        
        if not simulation_results:
            return {'success': False, 'message': 'No successful simulations'}
        
        # 결과 분석
        simulation_results = np.array(simulation_results)
        
        percentiles = {}
        for conf_level in confidence_levels:
            percentiles[f'p{int(conf_level * 100)}'] = np.percentile(simulation_results, conf_level * 100)
        
        return {
            'success': True,
            'num_simulations': len(simulation_results),
            'mean_return': np.mean(simulation_results),
            'std_return': np.std(simulation_results),
            'percentiles': percentiles,
            'var_95': np.percentile(simulation_results, 5),
            'cvar_95': np.mean(simulation_results[simulation_results <= np.percentile(simulation_results, 5)]),
            'all_results': simulation_results.tolist()
        }
    
    def _generate_random_scenario(self, returns_data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """랜덤 시나리오 생성"""
        scenario_data = {}
        
        for symbol, returns in returns_data.items():
            # 부트스트랩 샘플링
            n_samples = len(returns)
            bootstrap_returns = np.random.choice(returns, size=n_samples, replace=True)
            
            # 가격 시계열 생성
            initial_price = 100  # 임시 초기 가격
            prices = [initial_price]
            
            for ret in bootstrap_returns:
                prices.append(prices[-1] * (1 + ret))
            
            # 데이터프레임 생성
            dates = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
            df = pd.DataFrame({
                'Close': prices[1:],  # 첫 번째 가격 제외
                'Open': prices[:-1],
                'High': [max(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                'Low': [min(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                'Volume': [1000000] * (len(prices)-1)
            }, index=dates[1:])
            
            scenario_data[symbol] = df
        
        return scenario_data
