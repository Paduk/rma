import os
import pandas as pd
import re
import spacy
from Levenshtein import distance as lev_dist

# python -m spacy download en_core_web_sm
# import spacy
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# 키워드 추출 함수들
def extract_phones(text):
    return set(re.findall(r'\b\d{2,3}-\d{3,4}-\d{4}\b', text))

def extract_uris(text):
    return set(re.findall(r'https?://\S+', text))

def extract_emails(text):
    return set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text))

def extract_addresses(text):
    addr_pattern = r'\d{1,5}\s+[A-Za-z0-9.\s]+\s+(Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr)\b'
    return set(re.findall(addr_pattern, text, flags=re.I))

def extract_nouns(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"})

# 정확도 판단 함수 (정확히 같으면 1.0, 하나라도 틀리면 0.0)
def exact_match_score(ref, hyp):
    return 1.0 if ref == hyp else 0.0

# 문장 pair 평가 함수 (조건 적용)
def evaluate_sentence_pair(A, B):
    # 레벤슈타인 거리 먼저 계산
    semantic_distance = lev_dist(A, B)
    semantic_correct = semantic_distance <= 3

    if semantic_correct:
        # 거리 3 이하이면 무조건 맞다고 처리
        final_correct = True
        details = {
            "semantic_distance": semantic_distance,
            "semantic_correct": True,
            "keywords_required": False
        }
    else:
        # 거리 4 이상이면 키워드 검사 필수
        phones_a = extract_phones(A)
        phones_b = extract_phones(B)

        uris_a = extract_uris(A)
        uris_b = extract_uris(B)

        emails_a = extract_emails(A)
        emails_b = extract_emails(B)

        addresses_a = extract_addresses(A)
        addresses_b = extract_addresses(B)

        nouns_a = extract_nouns(A)
        nouns_b = extract_nouns(B)

        phone_match = exact_match_score(phones_a, phones_b)
        uri_match = exact_match_score(uris_a, uris_b)
        email_match = exact_match_score(emails_a, emails_b)
        addr_match = exact_match_score(addresses_a, addresses_b)
        noun_match = exact_match_score(nouns_a, nouns_b)

        keywords_all_correct = all([
            phone_match,
            uri_match,
            email_match,
            addr_match,
            noun_match
        ])

        final_correct = keywords_all_correct

        details = {
            "semantic_distance": semantic_distance,
            "semantic_correct": False,
            "keywords_required": True,
            "phone_match": bool(phone_match),
            "uri_match": bool(uri_match),
            "email_match": bool(email_match),
            "address_match": bool(addr_match),
            "noun_match": bool(noun_match),
        }

    return {
        "final_correct": final_correct,
        "details": details
    }



# 비교할 기본 디렉터리와 재작성된 파일 디렉터리
base_dir = "datasets/tc"
rew_dir = os.path.join(base_dir, "gemini-2.0-flash_rewrited")

# base_dir 내의 모든 .tsv 파일 목록 가져오기
file_list = [
    fn for fn in os.listdir(base_dir)
    if fn.endswith(".tsv") and os.path.isfile(os.path.join(base_dir, fn))
]

results = []

for file_name in sorted(file_list):
    path_a = os.path.join(base_dir, file_name)
    path_b = os.path.join(rew_dir, file_name)

    if not os.path.exists(path_b):
        #print(f"[경고] '{file_name}'에 대응되는 rewrited 파일을 찾을 수 없습니다.")
        continue

    df_a = pd.read_csv(path_a, sep="\t", dtype=str)
    df_b = pd.read_csv(path_b, sep="\t", dtype=str)

    if len(df_a) != len(df_b):
        print(f"[에러] '{file_name}'의 행 수가 다릅니다: {len(df_a)} vs {len(df_b)}")
        continue

    total = len(df_a)
    seq_a = df_a["rewrited_query"].fillna("")
    seq_b = df_b["rewrited_query"].fillna("")
    print(len(seq_b), len(df_b), '@@@@')

    exact_cnt = (seq_a == seq_b).sum()
    fuzzy_cnt = sum(1 for a, b in zip(seq_a, seq_b) if lev_dist(a, b) <= 3)

    # 100을 곱해서 퍼센트로, 소수점 둘째자리 포맷팅
    exact_pct = exact_cnt / total * 100
    fuzzy_pct = fuzzy_cnt / total * 100

    print(
        f"{file_name} -> "
        f"Exact Accuracy: {exact_pct:.2f}%, "
        f"Levenshtein≤3 Accuracy: {fuzzy_pct:.2f}%"
    )

    results.append({
        "file": file_name,
        "Exact (%)": exact_pct,
        "Lev≤3 (%)": fuzzy_pct
    })

# DataFrame으로 변환
results_df = pd.DataFrame(results).set_index("file")

# 소수점 둘째자리까지 반올림
results_df = results_df.round(2)

# 전체 평균 행 추가
avg = results_df.mean().round(2)
avg.name = "Average"
results_df = results_df.append(avg)

# TSV로 저장
results_df.to_csv("rma_logs/rewrited_score.tsv", sep="\t")

print("\n모든 파일 평가 결과를 'evaluation_results.tsv'에 저장했습니다.")
