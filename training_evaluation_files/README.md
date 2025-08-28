# LMNglyPred: Pre-trained Protein Language Model 임베딩을 이용한 인간 N-결합 글리코실화 부위 예측

## 모델 평가

프로그램은 아나콘다 버전 2020.07을 사용하여 실행되었으며, 동일 버전 설치를 권장합니다.

프로그램 개발 환경은 다음과 같습니다.
- **Python**: 3.8.3.final.0 (64비트)
- **OS**: Linux (5.8.0-38-generic)
- **Machine**: x86_64
- **Processor**: x86_64
- **Libraries**:
    - **pandas**: 1.0.5
    - **numpy**: 1.18.5
    - **pip**: 20.1.1
    - **scipy**: 1.4.1
    - **scikit-learn**: 0.23.1
    - **keras**: 2.4.3
    - **tensorflow**: 2.3.1

## 프로그램 실행 지침

아래 나열된 각 프로그램을 실행하려면, 해당 프로그램과 언급된 파일들을 **같은 디렉토리**에 놓고 실행하십시오.

1.  **GlycoBiology_NGlyDE_Original.ipynb** 실행 시 다음 파일들이 필요합니다.
    - `Undersampling_Glycobiology_NGLYDE_Final6947757.h5`
    - `Independent_Test_Set_Prot_T5_feature_Aug_12.txt`
    - `Subash_August_8_2022_NGlyDE_Prot_T5_feature.txt`

2.  **GlycoBiology_NGlyDE_90__Training_10__Indepedent_Testing.ipynb** 실행 시 다음 파일들이 필요합니다.
    - `NGlyDE_Prot_T5_Final.h5`
    - `Glycobiology_NGlyDE_Independent_Positive_202_Negative_100.csv`
    - `Glycobiology_NGlyDE_Training_Positive_1821_Negative_901.csv`

3.  **GlycoBiology_NGlycositeAtlas.ipynb** 실행 시 다음 파일들이 필요합니다.
    - `df_indepenent_test_again_done_that_has_unique_protein_and_unique_sequence.csv`
    - `Final_GlycoBiology_ANN_Glycobiology_ER_RSA(GA_Extracell_cellmem)187.h5`
    - `df_train_data_without_indepenent_test_and_protein.csv`

## ProtT5 기능 추출 프로그램

편의를 위해 **ProtT5**로부터 단백질 서열을 위한 기능 추출 프로그램(`analyze_Cell_Mem_ER_Extrac_Protein.py`)과 ProtT5 파일에서 1024개 특징 벡터를 추출하는 프로그램(`Feature Extraction Program from the generated files.ipynb`)을 제공합니다.

## 문의

추가 도움이 필요하시면, 다음 연락처로 문의하세요.
- **Dr. Subash Chandra Pakhrin**: pakhrins@uhd.edu
- **Dr. Dukka B. KC**: dbkc@mtu.edu