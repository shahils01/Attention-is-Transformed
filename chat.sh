python experiments/chat_tinystories.py \
  --checkpoint /scratch/shahils/lgma_runs/tinystories_lgma_multibase_b2/checkpoint_step_70000.pt \
  --data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-valid.txt \
  --device cuda:0