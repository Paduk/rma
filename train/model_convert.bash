ollama_path=/home/hj153lee/
model_dir=qwen3-rewrite
#checkpoint=2268
model=qwen3
#qwen25

#python model_merge.py --t1 ${model_dir}/checkpoint-${checkpoint} --t2 ${ollama_path}/${model_dir} --model ${model}&&
#python model_merge.py --t1 phi-4/rewrite/ --t2 /home/hj153lee/phi-4/rewrite/ --model phi4
python model_merge.py --t1 ../../${model_dir}/ --t2 ${ollama_path}/${model_dir} --model ${model}  &&
cd /home/hj153lee/llama.cpp &&

#python convert_hf_to_gguf.py ../phi-4/rewrite/ --outfile /home/hj153lee/phi-4-rewrite.gguf --outtype bf16
python convert_hf_to_gguf.py ../${model_dir} --outfile /home/hj153lee/${model_dir}.gguf --outtype bf16 &&
cd .. &&
echo "FROM ./${model_dir}.gguf" > ${model_dir}-modelfile
#ollama create ${model_dir} -f ${model_dir}-modelfile
#ollama create qwen2.5-rewrite -f qwen2.5-rewrite-modelfile