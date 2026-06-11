python experiments/chat_tinystories.py \
  --checkpoint /scratch/shahils/lgma_runs/large_tinystories_lgma_multibase_b2_h16/checkpoint_step_50000.pt \
  --data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-valid.txt \
  --device cuda:0