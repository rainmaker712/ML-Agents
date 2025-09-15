import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from .backtest_engine import BacktestEngine

class StrategyTester:
    """전략 테스터"""
    
    def __init__(self, initial_capital: float = 100000):
        self.backtest_engine = BacktestEngine(initial_capital)
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """기본 전략 등록"""
        self.strategies['buy_and_hold'] = self.buy_and_hold_strategy
        self.strategies['equal_weight'] = self.equal_weight_strategy
        self.strategies['momentum'] = self.momentum_strategy
        self.strategies['mean_reversion'] = self.mean_reversion_strategy
        self.strategies['rsi_strategy'] = self.rsi_strategy
        self.strategies['bollinger_bands'] = self.bollinger_bands_strategy
        self.strategies['macd_strategy'] = self.macd_strategy
    
    def test_strategy(self, strategy_name: str, data: Dict[str, pd.DataFrame],
                     parameters: Dict = None) -> Dict:
        """전략 테스트"""
        if strategy_name not in self.strategies:
            return {'success': False, 'message': f'Strategy {strategy_name} not found'}
        
        strategy_func = self.strategies[strategy_name]
        
        # 파라미터 적용
        if parameters:
            strategy_func = self._apply_parameters(strategy_func, parameters)
        
        # 백테스트 실행
        result = self.backtest_engine.run_backtest(data, strategy_func)
        
        if result['success']:
            result['strategy_name'] = strategy_name
            result['parameters'] = parameters or {}
        
        return result
    
    def compare_strategies(self, data: Dict[str, pd.DataFrame],
                          strategy_configs: List[Dict]) -> Dict:
        """여러 전략 비교"""
        results = {}
        
        for config in strategy_configs:
            strategy_name = config['name']
            parameters = config.get('parameters', {})
            
            result = self.test_strategy(strategy_name, data, parameters)
            results[strategy_name] = result
        
        # 비교 분석
        comparison = self._analyze_strategy_comparison(results)
        
        return {
            'success': True,
            'strategy_results': results,
            'comparison': comparison
        }
    
    def optimize_strategy_parameters(self, strategy_name: str,
                                   data: Dict[str, pd.DataFrame],
                                   parameter_ranges: Dict[str, List],
                                   optimization_metric: str = 'sharpe_ratio') -> Dict:
        """전략 파라미터 최적화"""
        if strategy_name not in self.strategies:
            return {'success': False, 'message': f'Strategy {strategy_name} not found'}
        
        # 파라미터 조합 생성
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        best_result = None
        best_metric_value = float('-inf')
        optimization_results = []
        
        for params in param_combinations:
            result = self.test_strategy(strategy_name, data, params)
            
            if result['success']:
                metric_value = result.get(optimization_metric, 0)
                optimization_results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0)
                })
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = result
                    best_result['optimal_parameters'] = params
        
        return {
            'success': True,
            'best_result': best_result,
            'best_metric_value': best_metric_value,
            'optimization_metric': optimization_metric,
            'all_results': optimization_results
        }
    
    def run_walk_forward_optimization(self, strategy_name: str,
                                    data: Dict[str, pd.DataFrame],
                                    parameter_ranges: Dict[str, List],
                                    train_period: int = 252,
                                    test_period: int = 63,
                                    step_size: int = 21) -> Dict:
        """워크 포워드 최적화"""
        if strategy_name not in self.strategies:
            return {'success': False, 'message': f'Strategy {strategy_name} not found'}
        
        # 전체 기간을 훈련/테스트 구간으로 분할
        all_dates = self.backtest_engine._get_common_dates(data)
        if len(all_dates) < train_period + test_period:
            return {'success': False, 'message': 'Insufficient data for walk-forward optimization'}
        
        results = []
        start_idx = 0
        
        while start_idx + train_period + test_period <= len(all_dates):
            # 훈련 기간
            train_dates = all_dates[start_idx:start_idx + train_period]
            train_data = self.backtest_engine._filter_data_by_dates(data, train_dates)
            
            # 테스트 기간
            test_dates = all_dates[start_idx + train_period:start_idx + train_period + test_period]
            test_data = self.backtest_engine._filter_data_by_dates(data, test_dates)
            
            # 훈련 기간에서 최적 파라미터 찾기
            optimization_result = self.optimize_strategy_parameters(
                strategy_name, train_data, parameter_ranges
            )
            
            if optimization_result['success'] and optimization_result['best_result']:
                # 최적 파라미터로 테스트 기간 실행
                optimal_params = optimization_result['best_result']['optimal_parameters']
                test_result = self.test_strategy(strategy_name, test_data, optimal_params)
                
                if test_result['success']:
                    results.append({
                        'train_start': train_dates[0],
                        'train_end': train_dates[-1],
                        'test_start': test_dates[0],
                        'test_end': test_dates[-1],
                        'optimal_parameters': optimal_params,
                        'test_return': test_result['total_return'],
                        'test_sharpe': test_result['sharpe_ratio'],
                        'test_max_dd': test_result['max_drawdown']
                    })
            
            start_idx += step_size
        
        if not results:
            return {'success': False, 'message': 'No valid optimization periods'}
        
        # 결과 분석
        test_returns = [r['test_return'] for r in results]
        test_sharpes = [r['test_sharpe'] for r in results]
        
        return {
            'success': True,
            'num_periods': len(results),
            'avg_return': np.mean(test_returns),
            'avg_sharpe': np.mean(test_sharpes),
            'return_std': np.std(test_returns),
            'sharpe_std': np.std(test_sharpes),
            'periods': results
        }
    
    # 기본 전략들
    def buy_and_hold_strategy(self, date, prices, positions, portfolio_value):
        """바이 앤 홀드 전략"""
        if not positions:  # 첫 거래
            # 균등 가중치로 매수
            num_stocks = len(prices)
            weight_per_stock = 1.0 / num_stocks
            return {symbol: weight_per_stock for symbol in prices.keys()}
        return None
    
    def equal_weight_strategy(self, date, prices, positions, portfolio_value):
        """균등 가중치 전략"""
        num_stocks = len(prices)
        weight_per_stock = 1.0 / num_stocks
        return {symbol: weight_per_stock for symbol in prices.keys()}
    
    def momentum_strategy(self, date, prices, positions, portfolio_value, 
                         lookback_period: int = 20, top_n: int = 5):
        """모멘텀 전략"""
        # 실제 구현에서는 과거 데이터를 사용하여 모멘텀 계산
        # 여기서는 간단한 랜덤 가중치 사용
        symbols = list(prices.keys())
        if len(symbols) <= top_n:
            weight_per_stock = 1.0 / len(symbols)
            return {symbol: weight_per_stock for symbol in symbols}
        
        # 상위 N개 종목 선택 (랜덤)
        selected_symbols = np.random.choice(symbols, size=top_n, replace=False)
        weight_per_stock = 1.0 / top_n
        return {symbol: weight_per_stock for symbol in selected_symbols}
    
    def mean_reversion_strategy(self, date, prices, positions, portfolio_value,
                               lookback_period: int = 20, threshold: float = 2.0):
        """평균 회귀 전략"""
        # 실제 구현에서는 과거 데이터를 사용하여 평균 회귀 계산
        # 여기서는 간단한 균등 가중치 사용
        symbols = list(prices.keys())
        weight_per_stock = 1.0 / len(symbols)
        return {symbol: weight_per_stock for symbol in symbols}
    
    def rsi_strategy(self, date, prices, positions, portfolio_value,
                    rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """RSI 전략"""
        # 실제 구현에서는 RSI 계산
        # 여기서는 간단한 균등 가중치 사용
        symbols = list(prices.keys())
        weight_per_stock = 1.0 / len(symbols)
        return {symbol: weight_per_stock for symbol in symbols}
    
    def bollinger_bands_strategy(self, date, prices, positions, portfolio_value,
                                period: int = 20, std_dev: float = 2.0):
        """볼린저 밴드 전략"""
        # 실제 구현에서는 볼린저 밴드 계산
        # 여기서는 간단한 균등 가중치 사용
        symbols = list(prices.keys())
        weight_per_stock = 1.0 / len(symbols)
        return {symbol: weight_per_stock for symbol in symbols}
    
    def macd_strategy(self, date, prices, positions, portfolio_value,
                     fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """MACD 전략"""
        # 실제 구현에서는 MACD 계산
        # 여기서는 간단한 균등 가중치 사용
        symbols = list(prices.keys())
        weight_per_stock = 1.0 / len(symbols)
        return {symbol: weight_per_stock for symbol in symbols}
    
    def _apply_parameters(self, strategy_func: Callable, parameters: Dict) -> Callable:
        """전략에 파라미터 적용"""
        def parameterized_strategy(date, prices, positions, portfolio_value):
            return strategy_func(date, prices, positions, portfolio_value, **parameters)
        return parameterized_strategy
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """파라미터 조합 생성"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _analyze_strategy_comparison(self, results: Dict) -> Dict:
        """전략 비교 분석"""
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            return {'error': 'No successful strategy results'}
        
        # 메트릭별 비교
        metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        comparison = {}
        
        for metric in metrics:
            metric_values = {name: result.get(metric, 0) for name, result in successful_results.items()}
            comparison[metric] = {
                'values': metric_values,
                'best': max(metric_values.items(), key=lambda x: x[1]) if metric != 'max_drawdown' else min(metric_values.items(), key=lambda x: x[1]),
                'worst': min(metric_values.items(), key=lambda x: x[1]) if metric != 'max_drawdown' else max(metric_values.items(), key=lambda x: x[1])
            }
        
        # 종합 순위
        strategy_scores = {}
        for name, result in successful_results.items():
            # 샤프 비율 기준으로 점수 계산
            sharpe = result.get('sharpe_ratio', 0)
            max_dd = result.get('max_drawdown', 1)
            score = sharpe * (1 - max_dd)  # 최대 낙폭 페널티
            strategy_scores[name] = score
        
        ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'metric_comparison': comparison,
            'strategy_ranking': ranked_strategies,
            'num_strategies': len(successful_results)
        }
    
    def add_custom_strategy(self, name: str, strategy_func: Callable):
        """커스텀 전략 추가"""
        self.strategies[name] = strategy_func
    
    def get_available_strategies(self) -> List[str]:
        """사용 가능한 전략 목록 반환"""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Dict:
        """전략 정보 반환"""
        if strategy_name not in self.strategies:
            return {'error': f'Strategy {strategy_name} not found'}
        
        # 실제 구현에서는 전략의 파라미터 정보 등을 반환
        return {
            'name': strategy_name,
            'description': f'Strategy: {strategy_name}',
            'parameters': {}  # 실제 구현에서는 전략의 파라미터 정보 반환
        }
