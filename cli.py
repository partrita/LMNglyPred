import click
import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras.models import load_model
from tqdm import tqdm
from typing import Any
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import os


def get_protT5_features(
    sequence: str,
    tokenizer: T5Tokenizer,
    pretrained_model: T5EncoderModel,
    device: torch.device,
) -> np.ndarray:
    # 희귀 아미노산을 X로 대체
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = [" ".join(sequence)]

    # 토큰화
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    # 메모리 효율적인 임베딩 추출
    with torch.no_grad():
        embedding = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # 즉시 텐서 메모리 해제
        del input_ids, attention_mask
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    seq_len = int((torch.tensor(ids["attention_mask"])[0] == 1).sum())
    seq_emd = embedding[0][: seq_len - 1]

    return seq_emd


@click.command()
@click.option(
    "--input-fasta-file",
    "-i",
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="입력 FASTA 파일 경로",
)
@click.option("--output-dir", "-o", default="output_chunks", help="출력 디렉토리 경로")
@click.option(
    "--model-path",
    "-m",
    default="models/NGlyDE_Prot_T5_Final.h5",
    show_default=True,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="Keras 모델 파일 경로",
)
@click.option(
    "--batch-size",
    "-b",
    default=32,
    show_default=True,
    type=int,
    help="배치 크기 (메모리에 따라 조정)",
)
@click.option(
    "--max-length",
    "-l",
    default=1000,
    show_default=True,
    type=int,
    help="최대 시퀀스 길이 (더 긴 시퀀스는 분할 처리)",
)
@click.option(
    "--chunk-size",
    "-c",
    default=10000,
    show_default=True,
    type=int,
    help="청크당 시퀀스 수",
)
def main(
    input_fasta_file: str,
    output_dir: str,
    model_path: str,
    batch_size: int,
    max_length: int,
    chunk_size: int,
) -> None:
    cutoff_threshold: float = 0.5

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    print("모델 로딩 중...")
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    pretrained_model: T5EncoderModel = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50"
    )

    # 예측 모델을 한 번만 로드
    combined_model: Any = load_model(model_path)
    print("모델 로딩 완료")

    gc.collect()

    device: torch.device = torch.device("cpu")
    pretrained_model = pretrained_model.to(device)
    pretrained_model = pretrained_model.eval()

    chunk_num = 0
    seq_count = 0
    results_df: pd.DataFrame = pd.DataFrame(
        columns=["prot_desc", "position", "site_residue", "probability", "prediction"]
    )

    print(f"청크 크기: {chunk_size}개 시퀀스씩 처리")

    for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
        prot_id: str = seq_record.id
        sequence: str = str(seq_record.seq)

        try:
            # 긴 시퀀스를 청크로 분할 처리
            if len(sequence) > max_length:
                process_long_sequence(
                    sequence,
                    prot_id,
                    tokenizer,
                    pretrained_model,
                    device,
                    combined_model,
                    cutoff_threshold,
                    results_df,
                    max_length,
                    batch_size,
                )
            else:
                process_sequence(
                    sequence,
                    prot_id,
                    tokenizer,
                    pretrained_model,
                    device,
                    combined_model,
                    cutoff_threshold,
                    results_df,
                    batch_size,
                )

            seq_count += 1

            # 청크 크기에 도달하면 저장하고 새로 시작
            if seq_count % chunk_size == 0:
                chunk_num += 1
                chunk_filename = os.path.join(
                    output_dir, f"chunk_{chunk_num:04d}_result.csv"
                )
                results_df.to_csv(chunk_filename, index=False)
                print(
                    f"\n청크 {chunk_num} 저장 완료: {chunk_filename} ({len(results_df)} 예측 결과)"
                )

                # 새로운 DataFrame으로 초기화
                results_df = pd.DataFrame(
                    columns=[
                        "prot_desc",
                        "position",
                        "site_residue",
                        "probability",
                        "prediction",
                    ]
                )
                gc.collect()

        except Exception as e:
            print(f"시퀀스 {prot_id} 처리 중 오류: {e}")
            seq_count += 1  # 오류가 있어도 카운트는 증가
            continue

    # 마지막 남은 결과 저장
    if len(results_df) > 0:
        chunk_num += 1
        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_num:04d}_result.csv")
        results_df.to_csv(chunk_filename, index=False)
        print(
            f"\n마지막 청크 {chunk_num} 저장 완료: {chunk_filename} ({len(results_df)} 예측 결과)"
        )

    print("\n전체 처리 완료!")
    print(f"- 총 처리된 시퀀스: {seq_count:,}")
    print(f"- 생성된 청크 파일: {chunk_num}개")
    print(f"- 출력 디렉토리: {output_dir}")


def process_sequence(
    sequence: str,
    prot_id: str,
    tokenizer,
    pretrained_model,
    device,
    combined_model,
    cutoff_threshold: float,
    results_df: pd.DataFrame,
    batch_size: int,
):
    """일반 크기 시퀀스 처리"""
    try:
        pt5_all: np.ndarray = get_protT5_features(
            sequence, tokenizer, pretrained_model, device
        )

        # N 잔기 위치와 특징을 배치로 수집
        n_positions = []
        n_features = []

        for index, amino_acid in enumerate(sequence):
            if amino_acid == "N":
                n_positions.append(index + 1)
                n_features.append(pt5_all[index])

        if n_features:
            # 배치 예측
            predictions = batch_predict(
                np.array(n_features), combined_model, batch_size
            )

            # 결과 저장
            for pos, pred in zip(n_positions, predictions):
                results_df.loc[len(results_df)] = [
                    prot_id,
                    pos,
                    "N",
                    1 - pred,
                    int(pred < cutoff_threshold),
                ]
    except Exception as e:
        print(f"시퀀스 {prot_id} 처리 중 오류: {e}")


def process_long_sequence(
    sequence: str,
    prot_id: str,
    tokenizer,
    pretrained_model,
    device,
    combined_model,
    cutoff_threshold: float,
    results_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
):
    """긴 시퀀스를 청크로 분할하여 처리"""
    seq_len = len(sequence)
    overlap = 50  # 청크 간 겹침

    for start in range(0, seq_len, max_length - overlap):
        end = min(start + max_length, seq_len)
        chunk = sequence[start:end]

        try:
            pt5_chunk = get_protT5_features(chunk, tokenizer, pretrained_model, device)

            n_positions = []
            n_features = []

            for index, amino_acid in enumerate(chunk):
                if amino_acid == "N":
                    actual_pos = start + index + 1
                    n_positions.append(actual_pos)
                    n_features.append(pt5_chunk[index])

            if n_features:
                predictions = batch_predict(
                    np.array(n_features), combined_model, batch_size
                )

                for pos, pred in zip(n_positions, predictions):
                    results_df.loc[len(results_df)] = [
                        prot_id,
                        pos,
                        "N",
                        1 - pred,
                        int(pred < cutoff_threshold),
                    ]
        except Exception as e:
            print(f"청크 {start}-{end} 처리 중 오류: {e}")

        # 청크 처리 후 메모리 정리
        gc.collect()


def batch_predict(features: np.ndarray, model, batch_size: int) -> np.ndarray:
    """배치 단위로 예측 수행"""
    predictions = []

    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.extend(batch_pred.flatten())

    return np.array(predictions)


if __name__ == "__main__":
    main()
