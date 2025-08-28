# LMNglyPred: 사전 훈련된 단백질 언어 모델의 임베딩을 사용한 인간 N-연결 당화 부위 예측

LMNglyPred는 사전 훈련된 단백질 언어 모델의 임베딩을 사용하여 인간 단백질의 N-연결 당화 부위를 예측하는 딥러닝 기반 도구입니다. 생물학적으로 관련된 N-X-[S/T] 시퀀온에 초점을 맞추어 고급 기계 학습을 활용하여 예측 정확도와 신뢰성을 향상시킵니다.


## 목차

- [저자](#저자)
- [개요](#개요)
- [설치](#설치)
- [사용법](#사용법)
  - [모델 평가](#모델-평가)
  - [자체 시퀀스 예측](#자체-시퀀스-예측)
  - [훈련 및 실험](#훈련-및-실험)
- [인용](#인용)
- [연락처](#연락처)

---

## 저자

- Subash C Pakhrin[^1]
- Suresh Pokharel[^2]
- Kiyoko F Aoki-Kinoshita[^3]
- Moriah R Beck[^4]
- Tarun K Dam[^5]
- Doina Caragea [^6]
- Dukka B KC [^2][^*]


[^1]: School of Computing, Wichita State University, Wichita, KS, USA  
Department of Computer Science and Engineering Technology, University of Houston-Downtown, Houston, TX, USA  
[^2]: Department of Computer Science, Michigan Technological University, Houghton, MI, USA  
[^3]: Glycan and Life Systems Integration Center (GaLSIC), Soka University, Tokyo, Japan  
[^4]: Department of Chemistry and Biochemistry, Wichita State University, Wichita, KS, USA  
[^5]: Laboratory of Mechanistic Glycobiology, Department of Chemistry, Michigan Technological University, Houghton, MI, USA  
[^6]: Department of Computer Science, Kansas State University, Manhattan, KS, USA  
[^*]: Corresponding Author: dbkc@mtu.edu


## 개요

단백질 N-연결 당화는 인간에서 중요한 번역 후 수정으로, N-X-[S/T] 모티프에서 발생합니다(여기서 X ≠ 프롤린). 그러나 모든 시퀀온이 당화되는 것은 아니므로 생물학적 연구에서 계산적 예측이 필수적입니다. LMNglyPred는 딥러닝과 단백질 언어 모델 임베딩을 활용하여 강력한 성능을 달성하며, 벤치마크 결과에서 민감도, 특이도, 매튜스 상관계수, 정밀도, 정확도가 각각 76.50%, 75.36%, 0.49, 60.99%, 75.74%를 보여줍니다[8][9][10].

## 설치

**Python 버전:** 3.9.7

### 저장소 복제:

```bash
gh repo clone partrita/LMNglyPred
```
### 필요한 라이브러리 설치:

의존성 제어를 위해 uv를 사용합니다.

```bash
uv sync
```

## 사용법

### 자체 시퀀스 예측

자체 시퀀스에서 N-연결 당화 부위를 예측하려면:

1. FASTA 시퀀스를 `input/sequence.fa`에 배치합니다.
2. 실행:
   ```bash
   # 기본 사용 (작은 파일)
  uv run cli.py -i input/seq_H.fasta -o output/result.csv

  # 큰 파일용 (메모리 절약 설정)
  uv run cli.py -i input/large_seq.fasta -o output/result.csv --batch-size 16 --max-length 500

  # 매우 큰 파일용 (최대 메모리 절약)
  uv run cli.py -i input/huge_seq.fasta -o output/result.csv --batch-size 8 --max-length 300
   ```
3. 결과는 `output/result.csv`에 나타납니다.


### 훈련 및 실험

- 훈련 및 추가 실험을 위한 데이터와 스크립트는 `training_and_evaluation` 폴더에서 찾을 수 있습니다.
- 자세한 지침은 해당 폴더 내의 `README.md`를 따르세요.


## 개선 사항

1. 메모리 효율성: 모델을 한 번만 로드하고 재사용, 배치 단위 예측으로 메모리 사용량 최적화, 명시적 메모리 정리 (`gc.collect()`, `torch.cuda.empty_cache()`)
2. 큰 시퀀스 처리: `--max-length` 옵션으로 긴 시퀀스를 청크로 분할, 청크 간 겹침(overlap)으로 경계 부분 누락 방지, 각 청크 처리 후 메모리 정리
3. 배치 처리: `--batch-size` 옵션으로 배치 크기 조정 가능, N 잔기들을 모아서 한 번에 예측
4. 오류 처리: 각 시퀀스/청크별 예외 처리, 하나의 시퀀스 실패가 전체를 중단시키지 않음

## 인용

연구에서 LMNglyPred를 사용하는 경우 다음과 같이 인용해 주세요:

> Subash C Pakhrin, PhD, et al., "LMNglyPred: prediction of human N-linked glycosylation sites using embeddings from a pre-trained protein language model," Glycobiology, Volume 33, Issue 5, May 2023, Pages 411–422.  
> [https://doi.org/10.1093/glycob/cwad033](https://doi.org/10.1093/glycob/cwad033)


## 연락처

질문이나 지원이 필요한 경우 연락처:  
- Dr. Subash Chandra Pakhrin: pakhrins@uhd.edu  
- Dr. Dukka B. KC: dbkc@mtu.edu
