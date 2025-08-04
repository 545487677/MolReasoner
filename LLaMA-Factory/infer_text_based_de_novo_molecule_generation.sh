pip install -e ".[torch,metrics]" --no-build-isolation
pip3 install deepspeed

export WANDB_DISABLED="true"
export SWANLAB_MODE="disabled"
export DISABLE_VERSION_CHECK=1


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python scripts/vllm_infer_selfies.py \
    --model_name_or_path ./MolReasoner/ckpt/grpo/grpo_text_based_de_novo_molecule_generation \
    --template qwen \
    --dataset text_based_de_novo_molecule_generation_test \
    --save_name xxxxx.json \
    --batch_size 512 \
    --max_new_tokens 2048 \
    --preprocessing_num_workers 8


# Note: Replace `xxxxx.json` with your desired output file name.
# Note: Replace model_name_or_path with the path to your trained model.