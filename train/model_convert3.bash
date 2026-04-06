ollama_path=/home/hj153lee/
path=/mnt/data/.cache/hj153lee/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a/
#/mnt/data/.cache/hj153lee/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/
#/mnt/data/.cache/hj153lee/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9/
#/mnt/data/.cache/hj153lee/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/
#/mnt/data/.cache/hj153lee/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a/
model_dir=qwen3-base
#checkpoint=2268
model=qwen3

cd /home/hj153lee/llama.cpp &&

#python convert_hf_to_gguf.py ../phi-4/rewrite/ --outfile /home/hj153lee/phi-4-rewrite.gguf --outtype bf16
python convert_hf_to_gguf.py ${path} --outfile /home/hj153lee/${model_dir}.gguf --outtype bf16 &&
cd .. &&
echo "FROM ./${model_dir}.gguf" > ${model_dir}-modelfile
#ollama create ${model_dir} -f ${model_dir}-modelfile
#ollama create qwen3-base -f qwen3-base-modelfile