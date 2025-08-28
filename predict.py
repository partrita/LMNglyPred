import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras.models import load_model
from tqdm import tqdm

# for ProtT5 model
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc


"""
파일 경로 및 기타 매개변수 정의
"""
input_fasta_file = "input/sequence.fasta"  # 테스트 시퀀스 로드
output_csv_file = "output/results.csv"
model_path = "training_evaluation_files/models/NGlyDE_Prot_T5_Final.h5"
cutoff_threshold = 0.5


"""
토크나이저와 사전 훈련된 ProtT5 모델 로드
"""
# SentencePiece transformers가 설치되지 않은 경우 설치
#!pip install -q SentencePiece transformers


tokenizer = T5Tokenizer.from_pretrained(
    "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
)
pretrained_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# pretrained_model = pretrained_model.half()
gc.collect()

# 디바이스 정의
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
pretrained_model = pretrained_model.to(device)
pretrained_model = pretrained_model.eval()


def get_protT5_features(sequence):
    """
    설명: 주어진 문자열에서 주어진 위치와 크기의 윈도우를 추출
         (더 많은 조건과 최적화 테스트 필요)
    입력:
        sequence (str): 길이 l의 문자열
    반환:
        tensor: l*1024
    """

    # 희귀 아미노산을 X로 대체
    sequence = re.sub(r"[UZOB]", "X", sequence)

    # 아미노산 사이에 공백 추가
    sequence = [" ".join(sequence)]

    # 설정 구성 및 특징 추출
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        embedding = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    # 길이 찾기
    seq_len = (attention_mask[0] == 1).sum()

    # 특징 선택
    seq_emd = embedding[0][: seq_len - 1]

    return seq_emd


# 결과 데이터프레임 생성
results_df = pd.DataFrame(
    columns=["prot_desc", "position", "site_residue", "probability", "prediction"]
)

for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = str(seq_record.seq)

    positive_predicted = []
    negative_predicted = []

    # 전체 시퀀스에 대한 protT5 추출 및 임시 데이터프레임에 저장
    pt5_all = get_protT5_features(sequence)

    # 시퀀스의 각 아미노산에 대한 임베딩 특징 및 윈도우 생성
    for index, amino_acid in enumerate(sequence):
        # 아미노산이 'N'인지 확인
        if amino_acid in ["N"]:
            # 인덱스가 0부터 시작하므로 사이트는 인덱스보다 1 크게 고려
            site = index + 1

            # 위에서 추출한 ProtT5 특징 가져오기
            X_test_pt5 = pt5_all[index]

            # 모델 로드
            combined_model = load_model(model_path)

            # 예측 결과
            y_pred = combined_model.predict(
                np.array(X_test_pt5.reshape(1, 1024)), verbose=0
            )[0][0]

            # results_df에 결과 추가
            results_df.loc[len(results_df)] = [
                prot_id,
                site,
                amino_acid,
                1 - y_pred,
                int(y_pred < cutoff_threshold),
            ]

# 결과 내보내기
print("결과 저장 중...")
results_df.to_csv(output_csv_file, index=False)
print("결과가 " + output_csv_file + "에 저장되었습니다")
