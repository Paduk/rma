# ACTION_VIEW_CONTACT - dial 100%
-> view_contacts의 device_response 에 전화번호를 주로 생성하게끔 변경 
# ACTION_EDIT_CONTACT - search_location, 3, 7, 70.0
-> location을 포함한 데이터를 생성하도록 example 추가, reason 수정
-> search location의 Description에 pysical address로 변경
# ACTION_EDIT_CONTACT - ACTION_NAVIGATE_TO_LOCATION, 3, 7, 70.0    
-> location을 포함한 데이터를 생성하도록 example 추가
# ACTION_EDIT_CONTACT - dial, 4, 6, 60.0
-> 현 상태 유지

python3 s2_generator.py --api apis/s3_apis_sub.jsonl --s datagen/deb_it1_s1_o3.jsonl --o datagen/deb_it1_s2.jsonl
python3 s3_generator.py --api apis/s3_apis_sub.jsonl --s datagen/deb_it1_s2.jsonl --o datagen/deb_it1_s3.jsonl

# get_contact_info - send_email, 4, 6, 60.0
# get_contact_info - search_location, 4, 6, 60.0
# get_contact_info - ACTION_NAVIGATE_TO_LOCATION, 4, 6, 60.0
# get_contact_info - dial, 5, 5, 50.0
# get_contact_info - send_message, 7, 3, 30.0
# ACTION_INSERT_CONTACT - send_email, 5, 5, 50.0    
# ACTION_INSERT_CONTACT - search_location, 3, 7, 70.0
# ACTION_INSERT_CONTACT - dial, 6, 4, 40.0
# ACTION_INSERT_EVENT - search_location, 6, 4, 40.0
# ACTION_INSERT_EVENT - ACTION_NAVIGATE_TO_LOCATION, 6, 4, 40.0