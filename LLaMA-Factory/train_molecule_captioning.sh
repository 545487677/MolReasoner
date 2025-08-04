pip install -e ".[torch,metrics]" --no-build-isolation
pip3 install deepspeed

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

export WANDB_DISABLED="true"
export SWANLAB_MODE="disabled"
CUDA_VISIBLE_DEVICfuES=0,1,2,3,4,5,6,7,8 FORCE_TORCHRUN=1 llamafactory-cli train  LLaMA-Factory/examples/train_full/train_molecule_captioning/sft.yml