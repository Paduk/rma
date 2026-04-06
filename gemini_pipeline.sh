#MODEL="gemini-2.5-flash"
MODEL="gemini-2.0-flash"
DATADIR="gemini_datagen"
API_FILE="apis/api_v3.0.1.jsonl"

# Turn 1
# python3 s1_generator.py --o ${DATADIR}/it1_s1.jsonl --api "${API_FILE}" --model $MODEL &&

# Turn 2
python3 batch_s2_generator.py --s ${DATADIR}/it1_s1.jsonl --o ${DATADIR}/it1_s2.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s2 --t1 ${DATADIR}/it1_s1.jsonl --t2 ${DATADIR}/it1_s2.jsonl --o ${DATADIR}/it1_s2_idxed.jsonl &&
python3 batch_s3_generator.py --s ${DATADIR}/it1_s2.jsonl --o ${DATADIR}/it1_s3.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s3 --t1 ${DATADIR}/it1_s2_idxed.jsonl --t2 ${DATADIR}/it1_s3.jsonl --o ${DATADIR}/it1_s3_idxed.jsonl --model $MODEL &&
python3 dataset_spliter.py --t_list ${DATADIR}/it1_s3_idxed.jsonl &&
python3 batch_s3_rewriter.py --s ${DATADIR}/it1_s3_idxed_dedup.jsonl --o ${DATADIR}/it1_s3_idxed_rewrite.jsonl &&
python3 batch_s4_generator.py --s ${DATADIR}/it1_s3_idxed_rewrite.jsonl --o ${DATADIR}/it1_s4.jsonl --api "${API_FILE}" --model $MODEL &&
python3 filling_datas.py --s1 ${DATADIR}/it1_s1_spare.jsonl --s2 ${DATADIR}/it1_s2_idxed.jsonl --o ${DATADIR}/it1_supplyments.jsonl &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it1_s3_idxed_rewrite.jsonl --t2 ${DATADIR}/it1_s4.jsonl --o ${DATADIR}/it2_s1_nonnr.jsonl --model $MODEL &&
python3 balancing_data.py --t1 ${DATADIR}/it2_s1_nonnr.jsonl --t2 ${DATADIR}/it1_supplyments.jsonl --o ${DATADIR}/it2_s1.jsonl --model $MODEL &&
python3 filter_every_interation.py --t ${DATADIR}/it2_s1.jsonl --o logs/it2_s1_log.jsonl --model $MODEL &&

#Turn 3
python3 batch_s2_generator.py --s ${DATADIR}/it2_s1_filtered.jsonl --o ${DATADIR}/it2_s2.jsonl --api "${API_FILE}" --model $MODEL && 
python3 dataset_integration.py --step s2 --t1 ${DATADIR}/it2_s1_filtered.jsonl --t2 ${DATADIR}/it2_s2.jsonl --o ${DATADIR}/it2_s2_idxed.jsonl &&
python3 batch_s3_generator.py --s ${DATADIR}/it2_s2.jsonl --o ${DATADIR}/it2_s3.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s3 --t1 ${DATADIR}/it2_s2_idxed.jsonl --t2 ${DATADIR}/it2_s3.jsonl --o ${DATADIR}/it2_s3_idxed.jsonl --model $MODEL &&
python3 dataset_spliter.py --t_list ${DATADIR}/it1_s3_idxed_rewrite.jsonl ${DATADIR}/it2_s3_idxed.jsonl &&
python3 batch_s3_rewriter.py --s ${DATADIR}/it2_s3_idxed_dedup.jsonl --o ${DATADIR}/it2_s3_idxed_rewrite.jsonl &&
python3 batch_s4_generator.py --s ${DATADIR}/it2_s3_idxed_rewrite.jsonl --o ${DATADIR}/it2_s4.jsonl --api "${API_FILE}" --model $MODEL &&
python3 filling_datas.py --s1 ${DATADIR}/it1_s1_spare.jsonl --s2 ${DATADIR}/it2_s2_idxed.jsonl --o ${DATADIR}/it2_supplyments.jsonl &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it2_s3_idxed_rewrite.jsonl --t2 ${DATADIR}/it2_s4.jsonl --o ${DATADIR}/it3_s1_nonnr.jsonl --model $MODEL &&
python3 balancing_data.py --t1 ${DATADIR}/it3_s1_nonnr.jsonl --t2 ${DATADIR}/it2_supplyments.jsonl --o ${DATADIR}/it3_s1.jsonl --model $MODEL &&
python3 filter_every_interation.py --t ${DATADIR}/it3_s1.jsonl --o logs/it3_s1_log_gem.jsonl --model $MODEL &&

# # Turn 4
python3 batch_s2_generator.py --s ${DATADIR}/it3_s1_filtered.jsonl --o ${DATADIR}/it3_s2.jsonl --api "${API_FILE}" --model $MODEL && 
python3 dataset_integration.py --step s2 --t1 ${DATADIR}/it3_s1_filtered.jsonl --t2 ${DATADIR}/it3_s2.jsonl --o ${DATADIR}/it3_s2_idxed.jsonl &&
python3 batch_s3_generator.py --s ${DATADIR}/it3_s2.jsonl --o ${DATADIR}/it3_s3.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s3 --t1 ${DATADIR}/it3_s2_idxed.jsonl --t2 ${DATADIR}/it3_s3.jsonl --o ${DATADIR}/it3_s3_idxed.jsonl --model $MODEL &&
python3 dataset_spliter.py --t_list ${DATADIR}/it1_s3_idxed_rewrite_dedup.jsonl ${DATADIR}/it2_s3_idxed_rewrite.jsonl ${DATADIR}/it3_s3_idxed.jsonl &&
python3 batch_s3_rewriter.py --s ${DATADIR}/it3_s3_idxed_dedup.jsonl --o ${DATADIR}/it3_s3_idxed_rewrite.jsonl &&
python3 batch_s4_generator.py --s ${DATADIR}/it3_s3_idxed_rewrite.jsonl --o ${DATADIR}/it3_s4.jsonl --api "${API_FILE}" --model $MODEL &&
python3 filling_datas.py --s1 ${DATADIR}/it1_s1_spare.jsonl --s2 ${DATADIR}/it3_s2_idxed.jsonl --o ${DATADIR}/it3_supplyments.jsonl &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it3_s3_idxed_rewrite.jsonl --t2 ${DATADIR}/it3_s4.jsonl --o ${DATADIR}/it4_s1_nonnr.jsonl --model $MODEL &&
python3 balancing_data.py --t1 ${DATADIR}/it4_s1_nonnr.jsonl --t2 ${DATADIR}/it3_supplyments.jsonl --o ${DATADIR}/it4_s1.jsonl --model $MODEL &&
python3 filter_every_interation.py --t ${DATADIR}/it4_s1.jsonl --o logs/it4_s1_log_gem.jsonl --model $MODEL &&

# # Turn 5
python3 batch_s2_generator.py --s ${DATADIR}/it4_s1_filtered.jsonl --o ${DATADIR}/it4_s2.jsonl --api "${API_FILE}" --model $MODEL && 
python3 dataset_integration.py --step s2 --t1 ${DATADIR}/it4_s1_filtered.jsonl --t2 ${DATADIR}/it4_s2.jsonl --o ${DATADIR}/it4_s2_idxed.jsonl &&
python3 batch_s3_generator.py --s ${DATADIR}/it4_s2.jsonl --o ${DATADIR}/it4_s3.jsonl --api "${API_FILE}" --model $MODEL &&
python3 dataset_integration.py --step s3 --t1 ${DATADIR}/it4_s2_idxed.jsonl --t2 ${DATADIR}/it4_s3.jsonl --o ${DATADIR}/it4_s3_idxed.jsonl --model $MODEL &&
python3 dataset_spliter.py --t_list ${DATADIR}/it1_s3_idxed_rewrite_dedup_dedup.jsonl ${DATADIR}/it2_s3_idxed_rewrite_dedup.jsonl ${DATADIR}/it3_s3_idxed_rewrite.jsonl ${DATADIR}/it4_s3_idxed.jsonl &&
python3 batch_s3_rewriter.py --s ${DATADIR}/it4_s3_idxed_dedup.jsonl --o ${DATADIR}/it4_s3_idxed_rewrite.jsonl &&
python3 batch_s4_generator.py --s ${DATADIR}/it4_s3_idxed_rewrite.jsonl --o ${DATADIR}/it4_s4.jsonl --api "${API_FILE}" --model $MODEL &&
python3 filling_datas.py --s1 ${DATADIR}/it1_s1_spare.jsonl --s2 ${DATADIR}/it4_s2_idxed.jsonl --o ${DATADIR}/it4_supplyments.jsonl &&
python3 dataset_integration.py --step s4 --t1 ${DATADIR}/it4_s3_idxed_rewrite.jsonl --t2 ${DATADIR}/it4_s4.jsonl --o ${DATADIR}/it5_s1_nonnr.jsonl --model $MODEL &&
python3 balancing_data.py --t1 ${DATADIR}/it5_s1_nonnr.jsonl --t2 ${DATADIR}/it4_supplyments.jsonl --o ${DATADIR}/it5_s1.jsonl --model $MODEL &&
python3 filter_every_interation.py --t ${DATADIR}/it5_s1.jsonl --o logs/it5_s1_log_gem.jsonl --model $MODEL
