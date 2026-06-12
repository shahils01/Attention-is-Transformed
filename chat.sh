python experiments/chat_tinystories.py \
  --checkpoint /scratch/shahils/lgma_runs/large_tinystories_lgma_residual_multibase_b2_h16/checkpoint_step_175000.pt \
  --data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-train.txt \
  --val_data_path /scratch/shahils/lgma_data/tinystory/TinyStoriesV2-GPT4-valid.txt \
  --device cuda:0 \
  --max_new_tokens 1200 \
  --temperature 0.8 \
  --top_k 20