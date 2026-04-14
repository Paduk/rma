from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILES = [
    ROOT / "datasets/train/it2_nonNR_train.tsv",
    ROOT / "datasets/train/it3_nonNR_train.tsv",
    ROOT / "datasets/train/it4_nonNR_train.tsv",
    ROOT / "datasets/train/it5_nonNR_train.tsv",
    ROOT / "datasets/train/it2_NR_train.tsv",
    ROOT / "datasets/train/additional/it3_complex_1_train.tsv",
    ROOT / "datasets/train/additional/it4_complex_1_train.tsv",
    ROOT / "datasets/train/additional/it4_complex_2_train.tsv",
    ROOT / "datasets/train/additional/it5_complex_1_train.tsv",
    ROOT / "datasets/train/additional/it5_complex_2_train.tsv",
    ROOT / "datasets/train/additional/it5_complex_3_train.tsv",
]

TC_FILES = [
    ROOT / "datasets/tc/scale/it2_nonNR_tc.tsv",
    ROOT / "datasets/tc/scale/it3_nonNR_tc.tsv",
    ROOT / "datasets/tc/scale/it4_nonNR_tc.tsv",
    ROOT / "datasets/tc/scale/it5_nonNR_tc.tsv",
    ROOT / "datasets/tc/scale/it3_complex_1_tc.tsv",
    ROOT / "datasets/tc/scale/it4_complex_1_tc.tsv",
    ROOT / "datasets/tc/scale/it4_complex_2_tc.tsv",
    ROOT / "datasets/tc/scale/it5_complex_1_tc.tsv",
    ROOT / "datasets/tc/scale/it5_complex_2_tc.tsv",
    ROOT / "datasets/tc/scale/it5_complex_3_tc.tsv",
]


def summarize(files):
    rows = []
    for path in files:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        missing_columns = {"query", "rewrited_query"} - set(df.columns)
        if missing_columns:
            raise ValueError(f"{path.relative_to(ROOT)} missing columns: {sorted(missing_columns)}")

        same_count = int((df["query"] == df["rewrited_query"]).sum())
        total_count = len(df)
        rows.append((path.relative_to(ROOT), total_count, same_count))
    return rows


def print_summary(title, rows):
    print(title)
    total_rows = 0
    total_same = 0
    for path, row_count, same_count in rows:
        pct = same_count / row_count * 100 if row_count else 0
        print(f"{path}\trows={row_count}\tquery_eq_rewrited={same_count}\tpercent={pct:.2f}%")
        total_rows += row_count
        total_same += same_count

    pct = total_same / total_rows * 100 if total_rows else 0
    print(f"TOTAL\trows={total_rows}\tquery_eq_rewrited={total_same}\tpercent={pct:.2f}%")


def main():
    print_summary("=== Train ===", summarize(TRAIN_FILES))
    print()
    print_summary("=== TC ===", summarize(TC_FILES))


if __name__ == "__main__":
    main()
