#!/usr/bin/env python3
"""DDP-capable teacher-forced training for shared WMT14 EN→DE assets."""
from __future__ import annotations

import argparse, json, os, time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import sentencepiece as spm
from torch.nn.parallel import DistributedDataParallel as DDP

from lgma.seq2seq import TokenBucketBatcher, WMT14Dataset, WMT14Transformer, collate_translation
from lgma.tracking import finish_wandb, init_wandb_run, log_wandb


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--attention', default='mha')
    p.add_argument('--d-model', type=int, default=512); p.add_argument('--ffn-dim', type=int, default=2048)
    p.add_argument('--num-encoder-layers', type=int, default=6); p.add_argument('--num-decoder-layers', type=int, default=6)
    p.add_argument('--num-heads', type=int, default=8); p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--num-kv-heads', type=int); p.add_argument('--num-generators', type=int, default=8)
    p.add_argument('--stabilize-generators', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fuse-base-qkv', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--fold-value-transform-into-output', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--sdpa-gqa-mode', choices=['auto', 'native', 'expand'], default='auto')
    p.add_argument('--generator-mixing', choices=['softmax', 'none'], default='softmax')
    p.add_argument('--theta-init', choices=['random_sphere', 'circle'], default='random_sphere')
    p.add_argument('--theta-init-scale', type=float, default=0.02); p.add_argument('--generator-init-scale', type=float, default=0.02)
    p.add_argument('--metric-beta', type=float, default=1.0); p.add_argument('--value-beta', type=float)
    p.add_argument('--base-dim', type=int); p.add_argument('--value-dim', type=int); p.add_argument('--num-base-heads', type=int, default=1)
    p.add_argument('--value-transform', default='none'); p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--max-source-length', type=int, default=256); p.add_argument('--max-target-length', type=int, default=256)
    p.add_argument('--share-all-embeddings', action='store_true', help='Tie EN/DE input embeddings and target output weights for joint-vocabulary Transformer-base.')
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--tokens-per-batch', type=int, default=0, help='Per-GPU padded source and target token cap; enables length bucketing.')
    p.add_argument('--bucket-size', type=int, default=4096)
    p.add_argument('--steps', type=int, required=True); p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--adam-beta1', type=float, default=0.9); p.add_argument('--adam-beta2', type=float, default=0.98)
    p.add_argument('--adam-eps', type=float, default=1e-9)
    p.add_argument('--lr-schedule', choices=['constant', 'inverse_sqrt'], default='constant')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100); p.add_argument('--eval-every', type=int, default=1000); p.add_argument('--eval-batches', type=int, default=32)
    p.add_argument('--save-every', type=int, default=5000); p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--greedy-bleu-every', type=int, default=0, help='Optional monitoring BLEU; final reporting uses beam search.')
    p.add_argument('--greedy-bleu-samples', type=int, default=128)
    p.add_argument('--wandb-project'); p.add_argument('--wandb-entity'); p.add_argument('--wandb-run-name'); p.add_argument('--wandb-group'); p.add_argument('--wandb-tags'); p.add_argument('--wandb-mode', default='online')
    return p


def batch_from_indices(dataset, indices, cfg, device):
    batch = collate_translation([dataset[int(i)] for i in indices], pad_id=cfg['pad_id'], bos_id=cfg['bos_id'], eos_id=cfg['eos_id'], max_source_length=cfg['max_source_length'], max_target_length=cfg['max_target_length'])
    return type(batch)(*(x.to(device, non_blocking=True) for x in (batch.source_ids, batch.source_padding_mask, batch.decoder_input_ids, batch.target_ids)))


def translation_loss(model, batch, pad_id, label_smoothing=0.0, reduction='mean'):
    """Go through DDP's forward wrapper so gradients are synchronized correctly."""
    logits = model(batch.source_ids, batch.decoder_input_ids, batch.source_padding_mask)
    return F.cross_entropy(logits.flatten(0, 1), batch.target_ids.flatten(), ignore_index=pad_id, label_smoothing=label_smoothing, reduction=reduction)


def target_token_count(batch, pad_id):
    return batch.target_ids.ne(pad_id).sum()


def learning_rate(step, base_lr, schedule, warmup_steps):
    if schedule == 'constant':
        return base_lr
    if warmup_steps <= 0:
        raise ValueError('inverse_sqrt schedule requires --warmup-steps > 0')
    return base_lr * min(step / warmup_steps, (warmup_steps / step) ** 0.5)


@torch.no_grad()
def evaluate(model, dataset, cfg, device, batches):
    # Rank-zero-only evaluation must bypass DDP's forward wrapper, which may
    # otherwise perform collectives while the remaining ranks continue training.
    base = model.module if isinstance(model, DDP) else model
    base.eval(); loss_sum = torch.zeros((), device=device); token_count = torch.zeros((), device=device, dtype=torch.long)
    for start in range(0, min(len(dataset), batches * cfg['batch_size']), cfg['batch_size']):
        batch = batch_from_indices(dataset, range(start, min(start + cfg['batch_size'], len(dataset))), cfg, device)
        loss_sum += translation_loss(base, batch, cfg['pad_id'], reduction='sum')
        token_count += target_token_count(batch, cfg['pad_id'])
    base.train()
    return (loss_sum / token_count.clamp_min(1)).item()


@torch.no_grad()
def greedy_bleu(model, dataset, cfg, device, tokenizer_path, samples):
    """Small, explicitly labelled greedy BLEU monitor for smoke validation."""
    import sacrebleu
    base = model.module if isinstance(model, DDP) else model
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    base.eval(); hypotheses, references = [], []
    for start in range(0, min(len(dataset), samples), cfg['batch_size']):
        raw = [dataset[i] for i in range(start, min(start + cfg['batch_size'], len(dataset)))]
        batch = collate_translation(raw, pad_id=cfg['pad_id'], bos_id=cfg['bos_id'], eos_id=cfg['eos_id'], max_source_length=cfg['max_source_length'], max_target_length=cfg['max_target_length'])
        source = batch.source_ids.to(device); generated = base.greedy_decode(source, bos_id=cfg['bos_id'], eos_id=cfg['eos_id'], max_length=cfg['max_target_length'])
        for hypothesis, (_, target) in zip(generated.cpu().tolist(), raw):
            hypothesis = [x for x in hypothesis[1:] if x not in (cfg['pad_id'], cfg['eos_id'])]
            reference = [int(x) for x in target if x != cfg['eos_id']]
            hypotheses.append(tokenizer.decode(hypothesis)); references.append(tokenizer.decode(reference))
    base.train()
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a').score


def main():
    a = args_parser().parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0')); local = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world > 1
    if distributed: dist.init_process_group('nccl'); torch.cuda.set_device(local)
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed + rank); torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((a.data_dir / 'manifest.json').read_text()); ids = manifest['tokenizer']['special_ids']
    cfg = {'pad_id': ids['pad'], 'bos_id': ids['bos'], 'eos_id': ids['eos'], 'max_source_length': a.max_source_length, 'max_target_length': a.max_target_length, 'batch_size': a.batch_size}
    train, valid = WMT14Dataset(a.data_dir, 'train'), WMT14Dataset(a.data_dir, 'validation')
    # All ranks need the same permutation before rank-strided sharding; adding
    # rank to this seed causes overlaps and omissions between workers.
    batcher = TokenBucketBatcher(train, tokens_per_batch=a.tokens_per_batch, max_source_length=a.max_source_length, max_target_length=a.max_target_length, seed=a.seed, rank=rank, world_size=world, bucket_size=a.bucket_size) if a.tokens_per_batch else None
    attention_kw = {
        'num_generators': a.num_generators,
        'generator_mixing': a.generator_mixing,
        'num_kv_heads': a.num_kv_heads,
        'theta_init': a.theta_init,
        'theta_init_scale': a.theta_init_scale,
        'generator_init_scale': a.generator_init_scale,
        'metric_beta': a.metric_beta,
        'value_beta': a.value_beta,
        'base_dim': a.base_dim,
        'value_dim': a.value_dim,
        'num_base_heads': a.num_base_heads,
        'value_transform': a.value_transform,
        'stabilize_generators': a.stabilize_generators,
        'fuse_base_qkv': a.fuse_base_qkv,
        'fold_value_transform_into_output': a.fold_value_transform_into_output,
        'sdpa_gqa_mode': a.sdpa_gqa_mode,
    }
    model = WMT14Transformer(vocab_size=manifest['tokenizer']['vocab_size'], d_model=a.d_model, ffn_dim=a.ffn_dim, num_encoder_layers=a.num_encoder_layers, num_decoder_layers=a.num_decoder_layers, num_heads=a.num_heads, head_dim=a.head_dim, attention_type=a.attention, dropout=a.dropout, max_source_length=a.max_source_length, max_target_length=a.max_target_length, pad_id=ids['pad'], share_all_embeddings=a.share_all_embeddings, **attention_kw).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(a.adam_beta1, a.adam_beta2), eps=a.adam_eps, weight_decay=a.weight_decay)
    start = 0
    if a.resume_checkpoint:
        checkpoint = torch.load(a.resume_checkpoint, map_location=device, weights_only=False); model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); start = checkpoint['step']
    if distributed: model = DDP(model, device_ids=[local], output_device=local)
    if rank == 0: a.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed: dist.barrier()
    run = init_wandb_run(project=a.wandb_project, entity=a.wandb_entity, name=a.wandb_run_name, group=a.wandb_group, tags=a.wandb_tags, mode=a.wandb_mode, output_dir=a.output_dir, config={'args':vars(a), 'manifest':manifest, 'parameters':parameter_count, 'cross_attention':'standard_mha_shared_across_methods'}) if rank == 0 else None
    generator = torch.Generator().manual_seed(a.seed + rank); model.train(); train_seconds_since_log = 0.0; last_log_step = start
    for step in range(start + 1, a.steps + 1):
        train_started = time.perf_counter()
        current_lr = learning_rate(step, a.lr, a.lr_schedule, a.warmup_steps)
        for group in optimizer.param_groups: group['lr'] = current_lr
        optimizer.zero_grad(set_to_none=True); micro_batches = []
        for _ in range(a.grad_accum_steps):
            indices = batcher.next_batch() if batcher is not None else torch.randint(len(train), (a.batch_size,), generator=generator)
            micro_batches.append(batch_from_indices(train, indices, cfg, device))
        global_tokens = sum((target_token_count(batch, cfg['pad_id']) for batch in micro_batches), start=torch.zeros((), device=device, dtype=torch.long))
        if distributed: dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
        local_loss_sum = torch.zeros((), device=device)
        for micro, batch in enumerate(micro_batches):
            sync = model.no_sync() if distributed and micro + 1 < a.grad_accum_steps else nullcontext()
            with sync, torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.precision == 'bf16' and device.type == 'cuda'):
                loss_sum = translation_loss(model, batch, cfg['pad_id'], a.label_smoothing, reduction='sum')
                # DDP averages gradients across ranks, hence the world-size
                # factor. This makes every non-padding target token contribute
                # equally despite variable batch sizes and sequence lengths.
                loss = loss_sum * (world / global_tokens.clamp_min(1))
            loss.backward(); local_loss_sum += loss_sum.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_seconds_since_log += time.perf_counter() - train_started
        if step % a.log_every == 0 or step == 1:
            if distributed: dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
            loss = local_loss_sum / global_tokens.clamp_min(1)
            if rank == 0:
                payload={'step':step,'train_loss':loss.item(),'steps_per_second':(step-last_log_step)/max(train_seconds_since_log,1e-6),'target_tokens_per_step':int(global_tokens.item()),'tokens_per_batch_cap':a.tokens_per_batch or None,'parameters':parameter_count,'lr':current_lr}; print(json.dumps(payload)); log_wandb(run,payload,step)
            train_seconds_since_log = 0.0; last_log_step=step
        rank_zero_work = step % a.eval_every == 0 or (a.greedy_bleu_every and step % a.greedy_bleu_every == 0) or step % a.save_every == 0
        if distributed and rank_zero_work:
            dist.barrier()
        if step % a.eval_every == 0 and rank == 0:
            value=evaluate(model, valid, cfg, device, a.eval_batches); payload={'step':step,'validation_loss':value}; print(json.dumps(payload)); log_wandb(run,payload,step)
        if a.greedy_bleu_every and step % a.greedy_bleu_every == 0 and rank == 0:
            value=greedy_bleu(model, valid, cfg, device, a.data_dir / manifest['tokenizer']['model'], a.greedy_bleu_samples)
            payload={'step':step,'validation_greedy_bleu':value,'validation_greedy_bleu_samples':a.greedy_bleu_samples}; print(json.dumps(payload)); log_wandb(run,payload,step)
        if step % a.save_every == 0 and rank == 0:
            base=model.module if isinstance(model,DDP) else model; torch.save({'step':step,'model':base.state_dict(),'optimizer':optimizer.state_dict(),'args':vars(a),'manifest':manifest}, a.output_dir/f'checkpoint_step_{step}.pt')
        if distributed and rank_zero_work:
            dist.barrier()
    if rank == 0: finish_wandb(run)
    if distributed: dist.destroy_process_group()

if __name__ == '__main__': main()
