#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.vla_eval import VLAActionPlanner, euler_xyz_to_rot6d, rot6d_to_quat_xyzw, single_arm_to_proprio20


EP_LEN = 720
NUM_SEQUENCES = 1000


def _lazy_calvin_imports():
    try:
        import hydra
        from omegaconf import OmegaConf
        from calvin_agent.evaluation.multistep_sequences import get_sequences
        from calvin_agent.evaluation.utils import (
            collect_plan,
            count_success,
            get_env_state_for_initial_condition,
        )
        from calvin_env.envs.play_table_env import get_env
    except Exception as exc:
        raise RuntimeError(
            "CALVIN evaluation requires calvin_agent, calvin_env, hydra, and omegaconf. "
            "Install CALVIN in this environment before running evaluation."
        ) from exc
    return {
        "hydra": hydra,
        "OmegaConf": OmegaConf,
        "get_sequences": get_sequences,
        "collect_plan": collect_plan,
        "count_success": count_success,
        "get_env_state_for_initial_condition": get_env_state_for_initial_condition,
        "get_env": get_env,
    }


class CALVINLocalPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cuda:0", action_chunk: int | None = None):
        self.planner = VLAActionPlanner(checkpoint=checkpoint, device=device, action_chunk=action_chunk)

    def reset(self, obs: dict) -> None:
        robot_obs = obs["robot_obs"]
        left = np.concatenate(
            [
                robot_obs[:3],
                euler_xyz_to_rot6d(robot_obs[3:6]),
                (robot_obs[-1:] > 0.0).astype(np.float32),
            ],
            axis=0,
        )
        self.planner.reset(single_arm_to_proprio20(left))

    def step(self, obs: dict, language_instruction: str):
        robot_obs = obs["robot_obs"]
        left = np.concatenate(
            [
                robot_obs[:3],
                euler_xyz_to_rot6d(robot_obs[3:6]),
                (robot_obs[-1:] > 0.0).astype(np.float32),
            ],
            axis=0,
        )
        action20 = self.planner.next_action_plan(
            images=[obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]],
            language_instruction=language_instruction,
            proprio=single_arm_to_proprio20(left),
        )
        action = action20[:10]
        gripper = 1 if action[9] < 0.8 else -1
        return action[:3], rot6d_to_quat_xyzw(action[3:9]), gripper


def save_video(path: str | Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if frames:
        imageio.mimsave(str(path), frames, fps=fps)


def rollout(env, model: CALVINLocalPolicy, oracle, subtask, annotations, plans, debug: bool):
    obs = env.get_obs()
    language = annotations[subtask][0].split("\n")[0].replace("\u2019", "'")
    start_info = env.get_info()
    frames = []
    for step in range(EP_LEN):
        action = model.step(obs, language)
        obs, _, _, info = env.step(action)
        main = obs["rgb_obs"]["rgb_static"]
        wrist = np.asarray(Image.fromarray(obs["rgb_obs"]["rgb_gripper"]).resize(main.shape[:2]))
        frames.append(np.concatenate([main, wrist], axis=1))
        if step == 0:
            try:
                plans[subtask].append(action)
            except Exception:
                pass
        if oracle.get_task_info_for_set(start_info, info, {subtask}):
            if debug:
                print(f"success: {subtask}")
            return True, frames, language
    if debug:
        print(f"failed: {subtask}")
    return False, frames, language


def evaluate_sequence(env, model, oracle, init_state, sequence, annotations, plans, debug, output_dir: Path, helpers):
    robot_obs, scene_obs = helpers["get_env_state_for_initial_condition"](init_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset(env.get_obs())
    success = 0
    for subtask in sequence:
        ok, frames, language = rollout(env, model, oracle, subtask, annotations, plans, debug)
        safe_language = language.replace("/", "_")
        save_video(output_dir / f"{safe_language}_{int(ok)}.mp4", frames)
        if ok:
            success += 1
        else:
            break
    return success


def evaluate_policy(
    checkpoint: str | Path,
    calvin_root: str | Path,
    output_dir: str | Path,
    eval_start: int,
    eval_end: int,
    device: str,
    action_chunk: int | None,
    debug: bool = False,
) -> list[int]:
    helpers = _lazy_calvin_imports()
    root = Path(calvin_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conf_dir = root / "ABC_D" / "validation"
    if not conf_dir.exists():
        conf_dir = root
    task_cfg = helpers["OmegaConf"].load(conf_dir / "new_playtable_tasks.yaml")
    oracle = helpers["hydra"].utils.instantiate(task_cfg)
    annotations = helpers["OmegaConf"].load(conf_dir / "new_playtable_validation.yaml")
    sequences = list(helpers["get_sequences"](NUM_SEQUENCES))
    env = helpers["get_env"](conf_dir, show_gui=False)
    model = CALVINLocalPolicy(checkpoint=checkpoint, device=device, action_chunk=action_chunk)
    plans = defaultdict(list)
    results = []
    for idx in tqdm(range(eval_start, eval_end), desc="CALVIN sequences"):
        init_state, sequence = sequences[idx]
        score = evaluate_sequence(env, model, oracle, init_state, sequence, annotations, plans, debug, output, helpers)
        results.append(score)
        summary = helpers["count_success"](results)
        payload = {"sequence": idx, "score": score, "success_by_length": summary, "total": sum(summary)}
        print(json.dumps(payload))
        with (output / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"scores": results, "success_by_length": helpers["count_success"](results)}, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser("LGMA VLA CALVIN Evaluation Client")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt from train_vla.py")
    parser.add_argument(
        "--calvin_root",
        default="ABC_D/validation",
        help="Path to CALVIN validation config dir, or a root containing ABC_D/validation",
    )
    parser.add_argument("--output_dir", default="logs/calvin")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    evaluate_policy(
        checkpoint=args.checkpoint,
        calvin_root=args.calvin_root,
        output_dir=args.output_dir,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        device=args.device,
        action_chunk=args.action_chunk,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
