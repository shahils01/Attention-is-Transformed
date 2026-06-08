# Dataset Download Scripts

These scripts prepare plain-text files for the current character-level LM
runner. Use scratch/project storage on Palmetto rather than `$HOME`.

Recommended setup:

```bash
module load anaconda3
source activate llava_video

mkdir -p $SCRATCH/lgma_data
mkdir -p $SCRATCH/hf_cache

export DATA_DIR=$SCRATCH/lgma_data
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
```

Install optional data dependencies:

```bash
pip install -e ".[data]"
```

Download TinyStories:

```bash
python dataset_downloads/download_tinystories.py
```

Download WikiText-103:

```bash
python dataset_downloads/download_wikitext103.py
```

Download a 50k-row C4 subset:

```bash
python dataset_downloads/download_c4_subset.py \
  --output_dir "$DATA_DIR/c4_subset" \
  --cache_dir "$HF_HOME" \
  --rows 50000
```

If you previously exported a bad cache path such as `/hf_cache`, either unset it
or pass `--cache_dir /scratch/shahils/hf_cache`. The scripts now let an explicit
`--cache_dir` override existing Hugging Face cache environment variables.

Run a text experiment:

```bash
python experiments/train_tinystories.py \
  --data_path $DATA_DIR/tinystories/TinyStoriesV2-GPT4-train.txt \
  --steps 1000
```
