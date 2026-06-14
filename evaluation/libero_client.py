#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import imageio
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.vla_eval import VLAActionPlanner, rot6d_to_axis_angle, rotmat_to_rot6d, single_arm_to_proprio20


LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}


def _lazy_libero_imports():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        raise RuntimeError(
            "LIBERO evaluation requires the LIBERO/robosuite environment. "
            "Install LIBERO in this environment before running evaluation."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def _flip_agentview(image: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(image, 0), 1)


class LIBEROLocalPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cuda:0", action_chunk: int | None = None):
        self.planner = VLAActionPlanner(checkpoint=checkpoint, device=device, action_chunk=action_chunk)

    def reset(self) -> None:
        self.planner.reset()

    def step(self, obs: dict, language_instruction: str) -> np.ndarray:
        rot6d = obs["robo_ori"] if np.asarray(obs["robo_ori"]).shape[-1] == 6 else rotmat_to_rot6d(obs["robo_ori"])
        left_proprio = np.concatenate([obs["robo_pos"], rot6d, np.array([0.0], dtype=np.float32)], axis=0)
        proprio = single_arm_to_proprio20(left_proprio)
        action20 = self.planner.next_action_plan(
            images=[
                _flip_agentview(obs["agentview_image"]),
                obs["robot0_eye_in_hand_image"],
            ],
            language_instruction=language_instruction,
            proprio=proprio,
        )
        left = action20[:10]
        axis_angle = rot6d_to_axis_angle(left[3:9])
        gripper = np.array([1.0 if left[9] > 0.5 else -1.0], dtype=np.float32)
        return np.concatenate([left[:3], axis_angle, gripper], axis=0).astype(np.float32)


class LIBEROEvaluator:
    def __init__(
        self,
        task_suite_name: str,
        eval_horizon: int,
        num_episodes: int,
        init_seed: int,
        act_type: str,
        output_dir: Path,
    ) -> None:
        benchmark, get_libero_path, OffScreenRenderEnv = _lazy_libero_imports()
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.get_libero_path = get_libero_path
        self.OffScreenRenderEnv = OffScreenRenderEnv
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[task_suite_name]
        self.task_suite_list = [self.benchmark_dict[name]() for name in self.task_list]
        self.eval_horizon = int(eval_horizon)
        self.num_episodes = int(num_episodes)
        self.init_seed = int(init_seed)
        self.act_type = act_type
        self.output_dir = output_dir / task_suite_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_env(self, task_suite, task_id: int, episode_id: int):
        task = task_suite.get_task(task_id)
        bddl_file = os.path.join(
            self.get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        env = self.OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(self.init_seed + episode_id + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[episode_id % init_states.shape[0]])
        for _ in range(10):
            obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        if self.act_type == "abs":
            for robot in env.env.robots:
                robot.controller.use_delta = False
        elif self.act_type != "rel":
            raise ValueError("act_type must be 'abs' or 'rel'")
        return env, task.language, obs

    def _write_result(self, payload: dict[str, float]) -> None:
        print(payload)
        with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _rollout(self, policy: LIBEROLocalPolicy, task_suite, task_id: int, episode_id: int) -> float:
        env, language, obs = self._init_env(task_suite, task_id, episode_id)
        frames = []
        success = False
        try:
            for _ in tqdm(range(self.eval_horizon), desc=language):
                robot = env.env.robots[0]
                obs["robo_ori"] = rotmat_to_rot6d(robot.controller.ee_ori_mat)
                obs["robo_pos"] = robot.controller.ee_pos
                action = policy.step(obs, language)
                frames.append(_flip_agentview(obs["agentview_image"]))
                obs, _, done, _ = env.step(action)
                if done:
                    success = True
                    break
        finally:
            env.close()
        video_name = f"task{task_id:03d}_ep{episode_id:03d}_{int(success)}.mp4"
        if frames:
            imageio.mimsave((self.output_dir / video_name).as_posix(), frames, fps=30)
        score = 1.0 if success else 0.0
        self._write_result({f"sim/{self.task_suite_name}/{language}": score})
        return score

    def evaluate(self, policy: LIBEROLocalPolicy) -> float:
        scores = []
        for task_suite in self.task_suite_list:
            for task_id in tqdm(range(len(task_suite.tasks)), desc=f"{self.task_suite_name} tasks"):
                for episode_id in range(self.num_episodes):
                    policy.reset()
                    scores.append(self._rollout(policy, task_suite, task_id, episode_id))
        summary = float(np.mean(scores)) if scores else 0.0
        self._write_result({f"sim_summary/{self.task_suite_name}/all": summary})
        return summary


def eval_libero(
    checkpoint: str | Path,
    output_dir: str | Path,
    task_suites: Iterable[str],
    num_episodes: int,
    init_seed: int,
    act_type: str,
    device: str,
    action_chunk: int | None,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy = LIBEROLocalPolicy(checkpoint=checkpoint, device=device, action_chunk=action_chunk)
    results = {}
    for suite_name in task_suites:
        evaluator = LIBEROEvaluator(
            task_suite_name=suite_name,
            eval_horizon=LIBERO_DATASETS_HORIZON[suite_name],
            num_episodes=num_episodes,
            init_seed=init_seed,
            act_type=act_type,
            output_dir=output,
        )
        results[suite_name] = evaluator.evaluate(policy)
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser("LGMA VLA LIBERO Evaluation Client")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt from train_vla.py")
    parser.add_argument("--output_dir", default="logs/libero")
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
        choices=sorted(LIBERO_DATASETS.keys()),
    )
    parser.add_argument("--eval_time", type=int, default=10, help="Episodes per task")
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--act_type", choices=["abs", "rel"], default="abs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action_chunk", type=int, default=None)
    args = parser.parse_args()

    results = eval_libero(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        task_suites=args.task_suites,
        num_episodes=args.eval_time,
        init_seed=args.init_seed,
        act_type=args.act_type,
        device=args.device,
        action_chunk=args.action_chunk,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
