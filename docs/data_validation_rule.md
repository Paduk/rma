검증 대상 컬럼:
- conversation_history
- ori_query
- ori_rewrited_query
- answer
- plan
- new_query
- rewrited_query
- query

검증 원칙은 아래 3가지만 봐줘.

1. Intent 보존
new_query, rewrited_query, query가 모두 기존 plan / answer['plan']과 같은 intent로 해석되어야 한다.
표현이 바뀌어도 action 자체가 바뀌면 안 된다.
예: play_video가 send_email처럼 보이거나, set_alarm이 show_alarm처럼 보이면 실패.

2. Arguments 추론 가능성
answer['arguments']의 모든 핵심 값은 conversation_history와 각 query 계열 문장으로부터 추론 가능해야 한다.
- new_query는 it/that/them/there 같은 지시어를 써도 되지만, conversation_history로 대상이 분명해야 한다.
- rewrited_query는 지시어를 실제 URI, 파일명, 연락처, 장소, 시간, 알람 ID 등으로 명시적으로 풀어야 한다.
- query는 ASR/typo noise가 있어도 answer arguments를 추론하는 데 방해되면 안 된다.
- URI, 파일명, 이메일, 전화번호, 날짜, 시간, duration, alarm_id 같은 protected value는 바뀌거나 손상되면 실패.

3. 자연스러운 발화 / 노이즈 품질
new_query는 실제 사용자가 말할 법한 자연스러운 구어체여야 한다.
rewrited_query는 자연스럽지만 명시적인 rewrite여야 한다.
query는 new_query와 같은 의미를 유지하면서 가벼운 ASR/타이포 노이즈만 허용한다.
오타가 intent, 대상, 시간, 파일명, 연락처, 슬롯 값을 바꾸면 실패다.
query가 완전히 다른 문장이거나 이해하기 어려우면 실패다.

출력 형식:
- 문제가 있는 row만 보고해줘.
- 각 문제는 line 번호 또는 unique_idx, 문제 컬럼, severity, 간단한 이유, 필요하면 수정 제안 1개를 포함해줘.
- severity는 FAIL / FIX / MINOR 중 하나로 표시해줘.
  - FAIL: intent나 argument가 바뀜
  - FIX: 의미는 대체로 맞지만 rewrite, 자연스러움, 명시성이 부족함
  - MINOR: 사소한 문체/노이즈 품질 문제
- 문제가 없으면 "전체 PASS"라고 말해줘.
