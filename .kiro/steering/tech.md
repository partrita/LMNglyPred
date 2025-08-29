# 기술 스택

## 핵심 기술
- **Python**: 3.9.7+ (.python-version에 명시)
- **딥러닝**: TensorFlow 2.9.1, Keras 2.9.0
- **ML 라이브러리**: scikit-learn 1.2.0, XGBoost 1.5.0
- **단백질 언어 모델**: Transformers 4.18.0, PyTorch 1.11.0
- **생물정보학**: Biopython (Bio 1.5.2)
- **데이터 처리**: pandas 1.5.0, numpy 1.23.5

## 패키지 관리
- **주요 도구**: 의존성 관리 및 가상 환경을 위한 `uv`
- **락 파일**: 재현 가능한 빌드를 위한 `uv.lock`
- **설정**: 프로젝트 메타데이터 및 의존성을 위한 `pyproject.toml`

## 주요 의존성
- **ProtT5 모델**: Rostlab/prot_t5_xl_uniref50 (Hugging Face Transformers 통해)
- **CLI 프레임워크**: Click 8.1.8+
- **진행률 추적**: tqdm 4.63.0
- **시각화**: matplotlib 3.5.1, seaborn 0.11.2

## 일반적인 명령어

### 환경 설정
```bash
# 의존성 설치
uv sync

# 가상 환경 활성화 (필요시)
source .venv/bin/activate
```

### 예측 실행
```bash
# 기본 예측
uv run cli.py -i input/sequence.fasta -o output/result.csv

# 대용량 파일용 메모리 최적화
uv run cli.py -i input/large_seq.fasta -o output/result.csv --batch-size 16 --max-length 500

# 최대 메모리 절약
uv run cli.py -i input/huge_seq.fasta -o output/result.csv --batch-size 8 --max-length 300
```

### 개발
```bash
# 레거시 예측 스크립트 실행
python predict.py

# 시퀀스 청크 처리
python process_chunks.py

# FASTA 파일 분할
python split_fasta.py
```

## 메모리 관리
- 모델을 한 번 로드하고 예측 전반에 걸쳐 재사용
- 메모리 사용량 최적화를 위한 배치 처리
- 명시적 가비지 컬렉션 및 CUDA 캐시 정리
- 구성 가능한 배치 크기 및 시퀀스 길이 제한