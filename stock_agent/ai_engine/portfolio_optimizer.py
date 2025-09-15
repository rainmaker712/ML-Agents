import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """포트폴리오 최적화 클래스"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% 무위험 수익률
    
    def optimize_portfolio(self, expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray,
                          risk_tolerance: str = 'medium',
                          target_return: float = None,
                          max_weight: float = 0.3) -> Dict:
        """포트폴리오 최적화"""
        
        n_assets = len(expected_returns)
        
        # 리스크 허용도에 따른 목표 수익률 설정
        if target_return is None:
            if risk_tolerance == 'low':
                target_return = np.mean(expected_returns) * 0.7
            elif risk_tolerance == 'medium':
                target_return = np.mean(expected_returns) * 0.9
            else:  # high
                target_return = np.mean(expected_returns) * 1.1
        
        # 제약 조건 설정
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        # 경계 조건
        bounds = [(0, max_weight) for _ in range(n_assets)]
        
        # 초기 가중치 (균등 가중)
        x0 = np.array([1/n_assets] * n_assets)
        
        # 목적 함수 (샤프 비율 최대화)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # 최대화를 위해 음수
        
        # 최적화 실행
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
        else:
            return {
                'weights': x0,
                'expected_return': np.dot(x0, expected_returns),
                'volatility': np.sqrt(np.dot(x0.T, np.dot(cov_matrix, x0))),
                'sharpe_ratio': 0,
                'success': False,
                'message': result.message
            }
    
    def optimize_risk_parity(self, cov_matrix: np.ndarray) -> Dict:
        """리스크 패리티 포트폴리오 최적화"""
        n_assets = cov_matrix.shape[0]
        
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contrib = weights * np.dot(cov_matrix, weights) / portfolio_vol
            return np.sum((risk_contrib - 1/n_assets) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            return {
                'weights': result.x,
                'success': True
            }
        else:
            return {
                'weights': x0,
                'success': False,
                'message': result.message
            }
    
    def optimize_minimum_variance(self, cov_matrix: np.ndarray) -> Dict:
        """최소 분산 포트폴리오 최적화"""
        n_assets = cov_matrix.shape[0]
        
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            return {
                'weights': result.x,
                'success': True
            }
        else:
            return {
                'weights': x0,
                'success': False,
                'message': result.message
            }
    
    def calculate_efficient_frontier(self, expected_returns: np.ndarray, 
                                   cov_matrix: np.ndarray,
                                   num_portfolios: int = 100) -> Dict:
        """효율적 프론티어 계산"""
        n_assets = len(expected_returns)
        
        # 목표 수익률 범위
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]
            x0 = np.array([1/n_assets] * n_assets)
            
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                portfolio_return = np.dot(result.x, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                efficient_portfolios.append({
                    'weights': result.x,
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                })
        
        return {
            'portfolios': efficient_portfolios,
            'max_sharpe': max(efficient_portfolios, key=lambda x: x['sharpe_ratio']),
            'min_volatility': min(efficient_portfolios, key=lambda x: x['volatility'])
        }
    
    def cluster_analysis(self, returns_data: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """클러스터 분석을 통한 섹터 그룹핑"""
        if returns_data.empty:
            return {'clusters': [], 'labels': []}
        
        # 수익률 데이터 정규화
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(returns_data.fillna(0))
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_data)
        
        # 클러스터별 특성 분석
        clusters = []
        for i in range(n_clusters):
            cluster_data = returns_data[cluster_labels == i]
            if not cluster_data.empty:
                cluster_mean_return = cluster_data.mean().mean()
                cluster_volatility = cluster_data.std().mean()
                cluster_sharpe = cluster_mean_return / cluster_volatility if cluster_volatility > 0 else 0
                
                clusters.append({
                    'cluster_id': i,
                    'stocks': cluster_data.columns.tolist(),
                    'mean_return': cluster_mean_return,
                    'volatility': cluster_volatility,
                    'sharpe_ratio': cluster_sharpe,
                    'size': len(cluster_data.columns)
                })
        
        return {
            'clusters': clusters,
            'labels': cluster_labels,
            'n_clusters': n_clusters
        }
    
    def monte_carlo_simulation(self, expected_returns: np.ndarray, 
                              cov_matrix: np.ndarray,
                              num_simulations: int = 10000) -> Dict:
        """몬테카를로 시뮬레이션"""
        n_assets = len(expected_returns)
        
        # 랜덤 포트폴리오 생성
        random_weights = np.random.dirichlet(np.ones(n_assets), num_simulations)
        
        # 포트폴리오 수익률과 변동성 계산
        portfolio_returns = np.dot(random_weights, expected_returns)
        portfolio_volatilities = np.sqrt(np.sum(random_weights * np.dot(cov_matrix, random_weights.T).T, axis=1))
        sharpe_ratios = (portfolio_returns - self.risk_free_rate) / portfolio_volatilities
        
        # 통계 계산
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_volatilities': portfolio_volatilities,
            'sharpe_ratios': sharpe_ratios,
            'mean_return': np.mean(portfolio_returns),
            'mean_volatility': np.mean(portfolio_volatilities),
            'mean_sharpe': np.mean(sharpe_ratios),
            'max_sharpe_idx': np.argmax(sharpe_ratios),
            'best_weights': random_weights[np.argmax(sharpe_ratios)]
        }
    
    def rebalance_portfolio(self, current_weights: np.ndarray,
                           target_weights: np.ndarray,
                           threshold: float = 0.05) -> Dict:
        """포트폴리오 리밸런싱"""
        weight_diff = target_weights - current_weights
        
        # 리밸런싱이 필요한 포지션 식별
        rebalance_positions = np.abs(weight_diff) > threshold
        
        # 거래 비용 고려 (간단한 모델)
        transaction_cost = 0.001  # 0.1%
        total_turnover = np.sum(np.abs(weight_diff[rebalance_positions]))
        estimated_cost = total_turnover * transaction_cost
        
        return {
            'new_weights': target_weights,
            'weight_changes': weight_diff,
            'rebalance_positions': rebalance_positions,
            'total_turnover': total_turnover,
            'estimated_cost': estimated_cost,
            'rebalance_needed': np.any(rebalance_positions)
        }
    
    def calculate_portfolio_metrics(self, weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray) -> Dict:
        """포트폴리오 지표 계산"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # VaR 계산 (95% 신뢰구간)
        portfolio_std = portfolio_volatility
        var_95 = portfolio_return - 1.645 * portfolio_std
        
        # CVaR 계산
        cvar_95 = portfolio_return - 2.33 * portfolio_std
        
        # 최대 낙폭 추정
        max_drawdown_estimate = portfolio_volatility * 2.5
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown_estimate': max_drawdown_estimate,
            'return_per_volatility': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        }
