# 프로젝트 구조

## 루트 디렉토리 레이아웃
```
├── cli.py                    # 메인 CLI 인터페이스 (권장)
├── predict.py               # 레거시 예측 스크립트
├── process_chunks.py        # 청크 처리 유틸리티
├── split_fasta.py          # FASTA 파일 분할 유틸리티
├── pyproject.toml          # 프로젝트 설정 및 의존성
├── requirements.txt        # 레거시 요구사항 (pyproject.toml 사용 권장)
├── uv.lock                 # 의존성 락 파일
└── README.md               # 프로젝트 문서 (한국어)
```

## 데이터 디렉토리
- **`input/`**: 입력 FASTA 시퀀스 파일
- **`output/`**: 예측 결과 (CSV 형식)
- **`models/`**: 사전 훈련된 모델 파일 (.h5 형식)
  - `NGlyDE_Prot_T5_Final.h5` - 메인 예측 모델
  - `Final_GlycoBiology_ANN_Glycobiology_ER_RSA(GA_Extracell_cellmem)187.h5`
  - `Undersampling_Glycobiology_NGLYDE_Final6947757.h5`

## 개발 및 연구
- **`notebooks/`**: 분석 및 실험용 Jupyter 노트북
- **`scripts/`**: 단백질 분석용 유틸리티 스크립트
- **`training_evaluation_files/`**: 완전한 훈련 및 평가 파이프라인
  - `codes/` - 훈련 노트북 및 스크립트
  - `data/` - 훈련 및 테스트 데이터셋
  - `models/` - 훈련된 모델 파일
  - `README.md` - 상세한 평가 지침

## 설정 및 환경
- **`.python-version`**: Python 버전 명시 (3.9.7)
- **`.venv/`**: 가상 환경 (uv로 관리)
- **`.kiro/`**: Kiro IDE 설정 및 steering 규칙

## 파일 명명 규칙
- **모델**: 버전/날짜 접미사가 있는 설명적 이름 사용
- **데이터 파일**: 파일명에 데이터셋 유형 및 샘플 수 포함
- **출력**: 배치 처리를 위한 타임스탬프 또는 청크 번호 사용
- **스크립트**: 설명적 액션 이름과 함께 snake_case 사용

## 주요 아키텍처 패턴
- **모듈형 CLI**: Click을 통해 cli.py로 메인 기능 노출
- **배치 처리**: 대용량 시퀀스 파일의 메모리 효율적 처리
- **모델 재사용**: 모델을 한 번 로드하고 여러 예측에서 재사용
- **청크 출력**: 대용량 데이터셋을 관리 가능한 청크로 분할
- **오류 격리**: 개별 시퀀스 실패가 배치 처리를 중단시키지 않음

## 데이터 흐름
1. 입력 FASTA 파일 → `input/`
2. 시퀀스 처리 → ProtT5 특징 추출
3. 모델 예측 → Keras/TensorFlow 모델
4. 결과 출력 → CSV 파일로 `output/`
5. 대용량 파일 → 번호가 매겨진 출력 파일로 청크 처리