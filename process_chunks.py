#!/usr/bin/env python3
"""
청크 파일들을 순차적으로 처리하고 결과를 병합하는 스크립트
"""

import click
import os
import glob
import pandas as pd
import subprocess
from pathlib import Path


@click.command()
@click.option(
    "--chunks-dir", "-c", default="chunks", help="청크 파일들이 있는 디렉토리"
)
@click.option("--output-file", "-o", required=True, help="최종 출력 CSV 파일")
@click.option("--batch-size", "-b", default=8, type=int, help="배치 크기")
@click.option("--max-length", "-l", default=3000, type=int, help="최대 시퀀스 길이")
def process_chunks(chunks_dir, output_file, batch_size, max_length):
    """청크 파일들을 순차적으로 처리"""

    # 청크 파일 목록 가져오기
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.fasta")))

    if not chunk_files:
        print(f"청크 파일을 찾을 수 없습니다: {chunks_dir}")
        return

    print(f"총 {len(chunk_files)}개 청크 파일 발견")

    # 결과 파일들을 저장할 디렉토리
    results_dir = "chunk_results"
    os.makedirs(results_dir, exist_ok=True)

    # 각 청크 처리
    for i, chunk_file in enumerate(chunk_files, 1):
        chunk_name = Path(chunk_file).stem
        result_file = os.path.join(results_dir, f"{chunk_name}_result.csv")

        print(f"\n[{i}/{len(chunk_files)}] 처리 중: {chunk_file}")

        # 이미 처리된 파일이 있으면 건너뛰기
        if os.path.exists(result_file):
            print(f"이미 처리됨: {result_file}")
            continue

        # CLI 명령 실행
        cmd = [
            "uv",
            "run",
            "cli.py",
            "-i",
            chunk_file,
            "-o",
            result_file,
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )  # 1시간 타임아웃

            if result.returncode == 0:
                print(f"성공: {result_file}")
            else:
                print(f"오류 발생: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"타임아웃: {chunk_file}")
        except Exception as e:
            print(f"예외 발생: {e}")

    # 결과 병합
    print("\n결과 파일들 병합 중...")
    merge_results(results_dir, output_file)


def merge_results(results_dir, output_file):
    """결과 CSV 파일들을 병합"""
    result_files = glob.glob(os.path.join(results_dir, "*_result.csv"))

    if not result_files:
        print("병합할 결과 파일이 없습니다.")
        return

    print(f"{len(result_files)}개 결과 파일 병합 중...")

    # 첫 번째 파일로 시작
    combined_df = pd.read_csv(result_files[0])
    print(f"첫 번째 파일: {len(combined_df)} 행")

    # 나머지 파일들 추가
    for result_file in result_files[1:]:
        try:
            df = pd.read_csv(result_file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"추가됨: {result_file} ({len(df)} 행)")
        except Exception as e:
            print(f"파일 읽기 오류: {result_file} - {e}")

    # 최종 결과 저장
    combined_df.to_csv(output_file, index=False)
    print(f"\n병합 완료: {output_file} (총 {len(combined_df)} 행)")


if __name__ == "__main__":
    process_chunks()
