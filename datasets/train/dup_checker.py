# dup_checker.py

import pandas as pd
import ast



def check_duplicates_and_shuffle(file_list, output_path='deduped_shuffled.tsv', random_state=42):
    """
    1) 각 파일별 레코드 수 출력
    2) 'conversation_history' 문자열을 list로 변환하여 turn_cnt 컬럼 생성
    3) 모든 파일을 합쳐서 데이터 전체를 섞음 (shuffle)
    4) 'query' 기준으로 중복 제거
    5) 중복 제거 후 turn_cnt 분포 출력
    6) 결과를 TSV로 저장 (turn_cnt 컬럼 제외)
    """
    dfs = []
    for path in file_list:
        # TSV 읽기 (모든 컬럼 문자열로)
        df = pd.read_csv(path, sep='\t', dtype=str)
        print(f"'{path}' has {len(df)} records")

        # conversation_history 문자열을 list로 변환
        df['conversation_history_list'] = df['conversation_history'].apply(ast.literal_eval)
        # turn count 계산
        df['turn_cnt'] = df['conversation_history_list'].apply(len)

        dfs.append(df)

    # 모든 데이터프레임 합치기
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nCombined total before shuffling: {len(combined)} records")

    # shuffle 전체 데이터
    combined_shuffled = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("Shuffled the combined DataFrame")

    # 중복 제거
    deduped = combined_shuffled.drop_duplicates(subset='query').reset_index(drop=True)
    print(f"After removing duplicates on 'query': {len(deduped)} records")

    # turn_cnt 분포 출력
    dist = deduped['turn_cnt'].value_counts().sort_index()
    print("\nDistribution of turn_cnt after deduplication:")
    for turn, count in dist.items():
        print(f"Turns={turn}: {count} records")

    # turn_cnt 컬럼 제외하고 저장
    output_df = deduped.drop(columns=['turn_cnt'])
    output_df = output_df.drop(columns=['conversation_history_list'])
    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"Deduplicated DataFrame saved to '{output_path}' (without turn_cnt column)")

    return deduped

if __name__ == "__main__":
    # 여기에 처리할 TSV 파일 경로들을 나열하세요.
    file_list = [        
        "it2_NR_train.tsv",
        "it3_NR_train.tsv",
        "it4_NR_train.tsv",
        "it5_NR_train.tsv",
        # ...
    ]
    
    deduped_df = check_duplicates_and_shuffle(file_list, output_path='weighted/deduped_NR.tsv')
    # 필요하다면 deduped_df.to_csv(...) 등으로 저장할 수 있습니다.
