#!/bin/bash
export VLLM_ATTENTION_BACKEND=XFORMERS
ray stop                         
sleep 1s 
export HYDRA_FULL_ERROR=1
pip install --force-reinstall psutil==5.9.8
pip install -U "ray[data,train,tune,serve,default]"
pip install EFGs
pip install swanlab
pip install --upgrade boto3 botocore
pip install rdkit tensorboard
pip install python-Levenshtein
pip install selfies

export RAY_BACKEND_LOG_LEVEL=error
export RAY_DEDUP_LOGS=0
export SWANLAB_API_KEY=xxxxx      # Note: Set this to your actual SWANLAB API key
export SWANLAB_LOG_DIR=xxx # Note: Set this to your desired log directory

TRAIN_BATCH_SIZES=(256)
LENGTH_TOKEN=(32768)
LRS=(1e-7)
ROLLOUT_NS=8
TEMPERATURES=(0.7)


# export RAY_DEBUG=legacy
# # start head node
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --ray-debugger-external --port 6380
# # start worker node
# ray start --address='0.0.0.0:6380' --ray-debugger-external


# Note:  data.train_files and data.val_files should be set to the correct paths for your dataset.
# Note: actor_rollout_ref.model.path should point to the pre-trained model checkpoint you want to use for training.
# Note: trainer.project_name and trainer.experiment_name should be set to your desired project and experiment names.



for length in "${LENGTH_TOKEN[@]}"; do
  for lr in "${LRS[@]}"; do
    for r_n in "${ROLLOUT_NS[@]}"; do
      for temp in "${TEMPERATURES[@]}"; do

        exp_name="text_guided_molecule_generation_${TRAIN_BATCH_SIZES}_lr${lr}_rollout${r_n}_temp${temp}_lenght${LENGTH_TOKEN}"
        log_path="xxxxxx/xxx/_${exp_name}.log"
        save_dir="xxxxx/text_guided_molecule_generation/${exp_name}"
        
        python3 -m verl.trainer.main_ppo \
          algorithm.adv_estimator=grpo \
          data.train_files=./MolReasoner/data/grpo/text_based_de_novo_molecule_generation/train.parquet \
          data.val_files=./MolReasoner/data/grpo/text_based_de_novo_molecule_generation/test.parquet \
          data.train_batch_size=$TRAIN_BATCH_SIZES \
          data.max_prompt_length=2048 \
          data.max_response_length=4096 \
          data.filter_overlong_prompts=True \
          data.truncation='error' \
          actor_rollout_ref.model.path=./MolReasoner/ckpt/sft/sft_text_based_de_novo_molecule_generation \
          actor_rollout_ref.actor.optim.lr=$lr \
          actor_rollout_ref.model.use_remove_padding=True \
          actor_rollout_ref.actor.use_dynamic_bsz=True \
          actor_rollout_ref.actor.ppo_mini_batch_size=64 \
          actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$LENGTH_TOKEN \
          actor_rollout_ref.actor.use_kl_loss=True \
          actor_rollout_ref.actor.kl_loss_coef=0.001 \
          actor_rollout_ref.actor.kl_loss_type=low_var_kl \
          actor_rollout_ref.actor.entropy_coeff=0 \
          actor_rollout_ref.model.enable_gradient_checkpointing=True \
          actor_rollout_ref.actor.fsdp_config.param_offload=False \
          actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
          actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LENGTH_TOKEN \
          actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
          actor_rollout_ref.rollout.name=vllm \
          actor_rollout_ref.rollout.temperature=$temp \
          actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
          actor_rollout_ref.rollout.n=$r_n \
          actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LENGTH_TOKEN \
          actor_rollout_ref.ref.fsdp_config.param_offload=False \
          algorithm.use_kl_in_reward=False \
          trainer.critic_warmup=0 \
          trainer.logger=['console','swanlab','tensorboard'] \
          trainer.project_name='text_guided_molecule_generation' \
          trainer.experiment_name=$exp_name \
          reward_model.exp_method=default \
          trainer.n_gpus_per_node=8 \
          trainer.nnodes=1 \
          trainer.save_freq=600 \
          trainer.test_freq=5 \
          trainer.total_epochs=15 2>&1 | tee $log_path

        cd $save_dir
        max_step=$(ls -d global_step_* | sed 's/global_step_//' | sort -n | tail -1)
        max_dir="global_step_$max_step"

        for d in global_step_*; do
          if [ "$d" != "$max_dir" ] && [ -d "$d" ]; then
            echo "Deleting $d"
            rm -rf "$d"
          fi
        done

      done
    done
  done
done


