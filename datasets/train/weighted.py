import pandas as pd
import ast
import os
import sys

def weighted_sample_tsv(file_path: str, fraction: float = 0.1, random_state: int = 42):
    # 1. TSV 읽기
    df = pd.read_csv(file_path, sep='\t', dtype=str)
    print(f"원본 데이터 크기: {len(df)}")
    # 2. answer 컬럼을 dict로 변환
    df['answer_dict'] = df['answer'].apply(ast.literal_eval)
    
    # 3. plan 컬럼 추출
    df['plan'] = df['answer_dict'].apply(lambda d: d.get('plan'))
    
    # 4. plan별로 fraction만큼 샘플링
    sampled = (
        df
        .groupby('plan', group_keys=False)
        .apply(lambda grp: grp.sample(frac=fraction, random_state=random_state))
        .drop(columns=['answer_dict'])  # 불필요한 중간 컬럼 제거
    )
    
    # 5. 저장할 파일명 생성
    base, ext = os.path.splitext(file_path)
    out_path = f"weighted/{base}_weighted{ext}"
    
    # 6. TSV로 저장
    sampled = sampled.drop(columns=['plan'])
    sampled.to_csv(out_path, sep='\t', index=False)
    print(f"샘플링된 데이터 크기: {len(sampled)}")
    print(f"샘플링된 데이터가 '{out_path}'에 저장되었습니다.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python weighted_sampling.py <input_file.tsv>")
        sys.exit(1)
    input_file = sys.argv[1]
    weighted_sample_tsv(input_file)