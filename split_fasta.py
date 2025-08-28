#!/usr/bin/env python3
"""
FASTA 파일을 작은 청크로 분할하는 스크립트
"""

import click
from Bio import SeqIO
import os


@click.command()
@click.option(
    "--input-file",
    "-i",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="입력 FASTA 파일",
)
@click.option("--output-dir", "-o", default="chunks", help="출력 디렉토리")
@click.option("--chunk-size", "-c", default=10000, type=int, help="청크당 시퀀스 수")
def split_fasta(input_file, output_dir, chunk_size):
    """FASTA 파일을 작은 청크로 분할"""

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    chunk_num = 0
    seq_count = 0
    output_file = None

    print(f"FASTA 파일 분할 시작: {input_file}")
    print(f"청크 크기: {chunk_size} 시퀀스")

    for record in SeqIO.parse(input_file, "fasta"):
        # 새 청크 파일 시작
        if seq_count % chunk_size == 0:
            if output_file:
                output_file.close()

            chunk_num += 1
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_num:04d}.fasta")
            output_file = open(chunk_filename, "w")
            print(f"청크 {chunk_num} 생성: {chunk_filename}")

        # 시퀀스 쓰기
        SeqIO.write(record, output_file, "fasta")
        seq_count += 1

    if output_file:
        output_file.close()

    print(f"분할 완료: 총 {seq_count}개 시퀀스를 {chunk_num}개 청크로 분할")


if __name__ == "__main__":
    split_fasta()
