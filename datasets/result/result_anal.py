import pandas as pd
import ast
import re


def extract_turn_from_col(col_name):
    inner_name = col_name.split('|', 1)[1] if '|' in col_name else col_name
    match = re.search(r"it(\d+)", inner_name)
    if not match:
        return None
    return int(match.group(1))


def extract_inner_name(col_name):
    return col_name.split('|', 1)[1] if '|' in col_name else col_name


def get_inner_file_order(inner_name):
    complex_match = re.search(r"complex_(\d+)", inner_name)
    if complex_match:
        return (0, int(complex_match.group(1)))
    return (1, inner_name)


def add_turn_separator_columns(df):
    if df is None or df.empty:
        return df

    pieces = []
    prev_turn = None
    spacer_count = 0

    for col in df.columns:
        current_turn = extract_turn_from_col(col)
        if prev_turn is not None and current_turn is not None and current_turn != prev_turn:
            spacer_count += 1
            spacer_label = " " * spacer_count
            pieces.append(pd.Series([""] * len(df), index=df.index, name=spacer_label))
        pieces.append(df[col])
        if current_turn is not None:
            prev_turn = current_turn

    return pd.concat(pieces, axis=1)

def build_pivot_for_files(file_list, selected_plans=None):
    """
    - file_list: 평가할 TSV 파일 경로 리스트
    - selected_plans: 관심 플랜 리스트. None이면 전체 플랜 사용
    """
    acc_dfs = []

    for path in file_list:
        df = pd.read_csv(path, sep='\t', dtype=str)
        df['gt_plan'] = df['gt'].apply(
            lambda x: ast.literal_eval(x).get('plan') if pd.notnull(x) else None
        )
        df['correct'] = df['all'].str.lower() == 'pass'

        if selected_plans is not None:
            df_sel = df[df['gt_plan'].isin(selected_plans)]
        else:
            df_sel = df

        if df_sel.empty:
            print(f"[경고] {path}에 해당 조건의 플랜 데이터가 없습니다.")
            continue

        pivot = df_sel.pivot_table(
            index='gt_plan',
            columns='file',
            values='correct',
            aggfunc='mean'
        ) * 100

        new_cols = {col: f"{path}|{col}" for col in pivot.columns}
        pivot = pivot.rename(columns=new_cols)
        acc_dfs.append(pivot)

    if not acc_dfs:
        return None

    result = pd.concat(acc_dfs, axis=1)
    file_order = {path: idx for idx, path in enumerate(file_list)}
    sorted_cols = sorted(
        result.columns,
        key=lambda col: (
            extract_turn_from_col(col) if extract_turn_from_col(col) is not None else 999,
            file_order.get(col.split('|', 1)[0], 999),
            get_inner_file_order(extract_inner_name(col)),
        )
    )
    result = result[sorted_cols]
    macro_avg = result.mean(axis=0, skipna=True)
    result.loc['Macro'] = macro_avg
    result = add_turn_separator_columns(result)
    return result


def compute_multi_tsv_inner_file(file_list, selected_plans):
    """
    - file_list: 평가할 TSV 파일 경로 리스트
    - selected_plans: 관심 플랜 리스트
    """
    full_result = build_pivot_for_files(file_list, selected_plans=None)
    selected_result = build_pivot_for_files(file_list, selected_plans=selected_plans)

    if full_result is None and selected_result is None:
        print("유효한 데이터가 하나도 없습니다.")
        return

    output_sections = []

    if full_result is not None:
        print("플랜별 × (외부TSV|내부파일) 정확도 (전체 플랜, 맨 아래 ‘Macro’는 컬럼별 평균):")
        print(full_result.fillna("N/A").to_string(float_format="{:.4f}".format))
        output_sections.append(("ALL_PLANS", full_result))

    if selected_result is not None:
        print("\n")
        print("플랜별 × (외부TSV|내부파일) 정확도 (selected_plans만, 맨 아래 ‘Macro’는 컬럼별 평균):")
        print(selected_result.fillna("N/A").to_string(float_format="{:.4f}".format))
        output_sections.append(("SELECTED_PLANS", selected_result))

    with open("result.tsv", "w", encoding="utf-8") as f:
        first = True
        for section_name, section_df in output_sections:
            if not first:
                f.write("\n")
            first = False
            f.write(f"[{section_name}]\n")
            section_df.to_csv(
                f,
                sep='\t',
                index=True,
                na_rep="N/A",
                float_format="%.2f"
            )


if __name__ == "__main__":
    file_list = [
        #"rma_complex.tsv",
        #"complex_rewrite_gemini.tsv",        
        #"complex_history_gemini.tsv"
        #"base_rewrite_gt.tsv",
        # "../manual/phi-history.tsv",
        # "../manual/phi-rewrite.tsv",
        "qwen3-base-260324.tsv",
        "qwen3-rewrite-260324.tsv",        
    ]
    selected_plans = [
        'ACTION_EDIT_ALARM', 'ACTION_EDIT_CONTACT', 'ACTION_EDIT_DOCUMENT',
        'ACTION_EDIT_VIDEO', 'ACTION_INSERT_CONTACT', 'ACTION_INSERT_EVENT',
        'ACTION_NAVIGATE_TO_LOCATION', 'ACTION_OPEN_CONTENT',
        'dial', 'play_music', 'play_video', 'search_location',
        'send_email', 'send_message'
    ]
    compute_multi_tsv_inner_file(file_list, selected_plans)
