#!/usr/bin/env python3
"""
미국 주식 포트폴리오 AI 에이전트 실행 스크립트
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='미국 주식 포트폴리오 AI 에이전트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run.py                    # Streamlit 웹 앱 실행
  python run.py --mode web         # Streamlit 웹 앱 실행
  python run.py --mode data        # 데이터 수집 실행
  python run.py --mode test        # 테스트 실행
  python run.py --mode setup       # 초기 설정 실행
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['web', 'data', 'test', 'setup'],
        default='web',
        help='실행 모드 선택 (기본값: web)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='웹 앱 포트 번호 (기본값: 8501)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='웹 앱 호스트 (기본값: localhost)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        run_web_app(args)
    elif args.mode == 'data':
        run_data_collection()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'setup':
        run_setup()

def run_web_app(args):
    """웹 앱 실행"""
    try:
        import streamlit as st
        from app import main as app_main
        
        print("🚀 미국 주식 포트폴리오 AI 에이전트를 시작합니다...")
        print(f"📍 웹 앱 주소: http://{args.host}:{args.port}")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        
        # Streamlit 앱 실행
        os.system(f"streamlit run app.py --server.port {args.port} --server.address {args.host}")
        
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("💡 다음 명령어로 설치하세요: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 웹 앱 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)

def run_data_collection():
    """데이터 수집 실행"""
    try:
        from services import DataService
        from models import DatabaseManager
        
        print("📊 데이터 수집을 시작합니다...")
        
        # 데이터베이스 초기화
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("✅ 데이터베이스 테이블이 생성되었습니다.")
        
        # 데이터 서비스 초기화
        data_service = DataService(db_manager)
        
        # 샘플 데이터 수집
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        print(f"📈 {len(symbols)}개 종목의 데이터를 수집합니다...")
        
        for symbol in symbols:
            print(f"  - {symbol} 데이터 수집 중...")
            result = data_service.collect_stock_data(symbol)
            
            if result['success']:
                print(f"    ✅ {symbol}: {result['data_points']}개 데이터 포인트")
            else:
                print(f"    ❌ {symbol}: {result['message']}")
        
        print("✅ 데이터 수집이 완료되었습니다.")
        
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("💡 다음 명령어로 설치하세요: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류가 발생했습니다: {e}")
        sys.exit(1)

def run_tests():
    """테스트 실행"""
    try:
        import unittest
        import subprocess
        
        print("🧪 테스트를 실행합니다...")
        
        # 테스트 디렉토리 확인
        test_dir = project_root / 'tests'
        if not test_dir.exists():
            print("⚠️  테스트 디렉토리가 없습니다. 기본 테스트를 실행합니다.")
            run_basic_tests()
            return
        
        # pytest 실행
        try:
            result = subprocess.run(['pytest', str(test_dir), '-v'], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode == 0:
                print("✅ 모든 테스트가 통과했습니다.")
            else:
                print("❌ 일부 테스트가 실패했습니다.")
                
        except FileNotFoundError:
            print("⚠️  pytest가 설치되지 않았습니다. 기본 테스트를 실행합니다.")
            run_basic_tests()
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)

def run_basic_tests():
    """기본 테스트 실행"""
    try:
        from utils import ValidationUtils, MathUtils, DateUtils
        from data_collectors import YFinanceCollector
        
        print("🔍 기본 기능 테스트를 실행합니다...")
        
        # 유효성 검사 테스트
        print("  - 유효성 검사 테스트...")
        assert ValidationUtils.validate_symbol('AAPL') == True
        assert ValidationUtils.validate_symbol('INVALID!') == False
        print("    ✅ 유효성 검사 테스트 통과")
        
        # 수학 유틸리티 테스트
        print("  - 수학 유틸리티 테스트...")
        import numpy as np
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        sharpe = MathUtils.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        print("    ✅ 수학 유틸리티 테스트 통과")
        
        # 날짜 유틸리티 테스트
        print("  - 날짜 유틸리티 테스트...")
        from datetime import datetime
        assert DateUtils.validate_date('2024-01-01') == True
        assert DateUtils.validate_date('invalid') == False
        print("    ✅ 날짜 유틸리티 테스트 통과")
        
        print("✅ 모든 기본 테스트가 통과했습니다.")
        
    except Exception as e:
        print(f"❌ 기본 테스트 실행 중 오류가 발생했습니다: {e}")

def run_setup():
    """초기 설정 실행"""
    try:
        print("⚙️  초기 설정을 시작합니다...")
        
        # 1. 환경 변수 파일 생성
        env_file = project_root / '.env'
        if not env_file.exists():
            print("📝 .env 파일을 생성합니다...")
            env_content = """# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
OPENAI_API_KEY=your_openai_key_here

# Database
DATABASE_URL=sqlite:///stock_agent.db

# App Settings
DEBUG=True
LOG_LEVEL=INFO
"""
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print("✅ .env 파일이 생성되었습니다.")
        else:
            print("ℹ️  .env 파일이 이미 존재합니다.")
        
        # 2. 데이터베이스 초기화
        print("🗄️  데이터베이스를 초기화합니다...")
        from models import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("✅ 데이터베이스가 초기화되었습니다.")
        
        # 3. 로그 디렉토리 생성
        log_dir = project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        print("📁 로그 디렉토리가 생성되었습니다.")
        
        # 4. 설정 파일 검증
        print("🔍 설정 파일을 검증합니다...")
        from config import Config
        
        if not Config.ALPHA_VANTAGE_API_KEY or Config.ALPHA_VANTAGE_API_KEY == 'your_alpha_vantage_key_here':
            print("⚠️  Alpha Vantage API 키를 설정해주세요.")
        
        if not Config.NEWS_API_KEY or Config.NEWS_API_KEY == 'your_news_api_key_here':
            print("⚠️  News API 키를 설정해주세요.")
        
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == 'your_openai_key_here':
            print("⚠️  OpenAI API 키를 설정해주세요.")
        
        print("✅ 초기 설정이 완료되었습니다.")
        print("💡 .env 파일에서 API 키를 설정한 후 다시 실행하세요.")
        
    except Exception as e:
        print(f"❌ 초기 설정 중 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
