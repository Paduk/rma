import pandas as pd
import ast

def evaluate_tsv_performance(input_tsv):
    # 1. TSV 읽기
    df = pd.read_csv(input_tsv, sep='\t', dtype=str)

    # 2. 'all' 컬럼을 기준으로 correct 여부 계산
    df['correct'] = df['all'].str.lower() == 'pass'

    # 3. gt 컬럼에서 plan 값 추출
    df['gt_plan'] = df['gt'].apply(lambda s: ast.literal_eval(s)['plan'])

    # 4. 전체 성능 (전체 데이터 정확도)
    overall_acc = df['correct'].mean() * 100

    # 5. 전체 Macro 정확도 (플랜별 정확도의 단순 평균)
    macro_acc = df.groupby('gt_plan')['correct'].mean().mean() * 100

    # 6. 파일별 전체 성능
    file_overall = (
        df.groupby('file')['correct']
          .mean()
          .mul(100)
          .round(2)
          .rename('file_accuracy_pct')
    )

    # 7. 파일별 Macro 정확도 (파일 내 플랜별 정확도의 단순 평균)
    file_macro = (
        df.groupby(['file', 'gt_plan'])['correct']
          .mean()
          .groupby('file')
          .mean()
          .mul(100)
          .round(2)
          .rename('file_macro_accuracy_pct')
    )

    # 8. 플랜별 파일 정확도 (pivot)
    plan_file_pivot = (
        df.groupby(['gt_plan', 'file'])['correct']
          .mean()
          .mul(100)
          .round(2)
          .rename('accuracy_pct')
          .reset_index()
          .pivot(index='gt_plan', columns='file', values='accuracy_pct')
          .fillna(0)
    )
    # 오른쪽에 plan별 평균 성능 열 추가
    plan_file_pivot['plan_avg_accuracy_pct'] = plan_file_pivot.mean(axis=1).round(2)

    # 9. 결과 출력
    print(f"전체 데이터 정확도: {overall_acc:.2f}%")
    print(f"전체 Macro 정확도: {macro_acc:.2f}%\n")

    print("파일별 전체 정확도:")
    print(file_overall.to_string(), "\n")

    print("파일별 Macro 정확도:")
    print(file_macro.to_string(), "\n")

    print("플랜별 파일 정확도 (pivot with plan avg):")
    print(plan_file_pivot.to_string(), "\n")

    # 10. TSV로 저장
    file_overall.to_csv('tmp/file_overall_accuracy.tsv', sep='	', header=True, index=True)
    file_macro.to_csv('tmp/file_macro_accuracy.tsv', sep='	', header=True, index=True)
    # 플랜별 파일 정확도 pivot 테이블 저장 (인덱스=플랜 포함)
    plan_file_pivot.to_csv('tmp/plan_file_accuracy_pivot.tsv', sep='	', header=True, index=True)

    print("▶ file_overall_accuracy.tsv, file_macro_accuracy.tsv, plan_file_accuracy_pivot.tsv 로 저장 완료")

if __name__ == "__main__":
    evaluate_tsv_performance('logs/it4_complex_weighted_history.tsv')
