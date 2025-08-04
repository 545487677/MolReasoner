# Tested with 2 & 4 GPUs

set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_gemma_2b.sh <nproc_per_node>  [other_configs...]"
#     exit 1
# fi

nproc_per_node=2
save_path=xxx

# Shift the arguments so $@ refers to the rest
shift 2


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=xxx \
    data.val_files=xxx \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=xxxx \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen2_5_7b-it \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@