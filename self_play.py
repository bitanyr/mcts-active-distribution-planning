"""Train a new policy/value model from random initialization.

Old checkpoints are intentionally neither loaded nor resumed.
MCTS uses neural surrogate values during tree search, while the final
placement of every episode is evaluated by the physical optimization
model.
"""

import argparse
import csv
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from core.mcts import MCTS
from core.network import ADNDeepNet
from core.replay_buffer import ReplayBuffer
from data.devices import (
    MAX_LOAD_SHEDDING_MWH_REP,
    MAX_RES_CURTAILMENT_MWH_REP,
)
from env.aps_env import ActivePlanningEnv
from optimization.solver import evaluate_placement


ARCHITECTURE_VERSION = "corrected-v5"


LOG_FIELDS = (
    "episode",
    "seed",
    "moves",
    "selected_stop",
    "terminal_value",
    "solver_ok",
    "compliant",
    "sample_used",
    "economic_cost",
    "cost_change_percent_vs_base",
    "objective_cost",
    "penalty_cost",
    "placement",
    "placement_ieee",
    "termination_condition",
    "failure_reasons",
    "max_cone_abs_gap",
    "nonexact_cone_points",
    "min_voltage_pu",
    "res_curtailment_mwh_representative",
    "load_shedding_mwh_representative",
    "valid_episodes_cumulative",
    "invalid_episodes_cumulative",
    "replay_samples",
    "optimizer_updates_cumulative",
    "physics_solves_cumulative",
    "episode_elapsed_seconds",
    "loss_total",
)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def performance_target(result):
    summary = result["compliance"]["summary"]

    res = float(
        summary.get(
            "res_curtailment_mwh_representative",
            float("inf"),
        )
    )

    load = float(
        summary.get(
            "load_shedding_mwh_representative",
            float("inf"),
        )
    )

    res_target = (
        1.0
        if not np.isfinite(res)
        else float(
            np.clip(
                res / MAX_RES_CURTAILMENT_MWH_REP,
                0.0,
                1.0,
            )
        )
    )

    load_target = (
        1.0
        if not np.isfinite(load)
        else float(
            np.clip(
                load / MAX_LOAD_SHEDDING_MWH_REP,
                0.0,
                1.0,
            )
        )
    )

    return np.array(
        [res_target, load_target],
        dtype=np.float32,
    )


def validate_training_arguments(
    episodes,
    simulations,
    max_moves,
    batch_size,
    updates_per_episode,
    checkpoint_interval,
):
    if int(episodes) < 1:
        raise ValueError(
            "episodes must be at least 1."
        )

    if int(simulations) < 1:
        raise ValueError(
            "simulations must be at least 1."
        )

    if int(max_moves) < 1:
        raise ValueError(
            "max_moves must be at least 1."
        )

    if int(batch_size) < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if int(updates_per_episode) < 0:
        raise ValueError(
            "updates_per_episode cannot be negative."
        )

    if int(checkpoint_interval) < 0:
        raise ValueError(
            "checkpoint_interval cannot be negative."
        )


def build_checkpoint_payload(
    *,
    network,
    optimizer,
    seed,
    episode,
    configured_episodes,
    valid_episodes,
    invalid_episodes,
    replay_samples,
    optimizer_updates,
    base_economic_cost,
    simulations,
    max_moves,
    batch_size,
    updates_per_episode,
    stop_prior_floor,
):
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "seed": int(seed),
        "episode_completed": int(episode),
        "configured_episodes": int(
            configured_episodes
        ),
        "valid_episodes": int(valid_episodes),
        "invalid_episodes": int(
            invalid_episodes
        ),
        "replay_samples": int(replay_samples),
        "optimizer_updates": int(
            optimizer_updates
        ),
        "base_reference_economic_cost": float(
            base_economic_cost
        ),
        "simulations": int(simulations),
        "max_moves": int(max_moves),
        "batch_size": int(batch_size),
        "updates_per_episode": int(
            updates_per_episode
        ),
        "stop_prior_floor": float(
            stop_prior_floor
        ),
        "exact_terminal_during_mcts": False,
        "state_dict": network.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "resumable_with_exact_replay_state": False,
    }


def save_checkpoint_atomic(path, payload):
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(path)


def train_from_scratch(
    episodes=400,
    simulations=500,
    max_moves=32,
    seed=0,
    batch_size=32,
    updates_per_episode=4,
    output_dir="models",
    checkpoint_interval=10,
    stop_prior_floor=0.05,
):
    validate_training_arguments(
        episodes=episodes,
        simulations=simulations,
        max_moves=max_moves,
        batch_size=batch_size,
        updates_per_episode=updates_per_episode,
        checkpoint_interval=checkpoint_interval,
    )

    episodes = int(episodes)
    simulations = int(simulations)
    max_moves = int(max_moves)
    seed = int(seed)
    batch_size = int(batch_size)
    updates_per_episode = int(
        updates_per_episode
    )
    checkpoint_interval = int(
        checkpoint_interval
    )
    stop_prior_floor = float(
        stop_prior_floor
    )

    set_global_seed(seed)

    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    log_path = (
        output
        / (
            f"training_{ARCHITECTURE_VERSION}"
            f"_seed{seed}_{stamp}.csv"
        )
    )

    env = ActivePlanningEnv()

    base = evaluate_placement(
        env.model,
        {},
        hard_verify=False,
    )

    base_is_valid = bool(
        base["solver_ok"]
        and base["is_compliant"]
        and np.isfinite(
            base["economic_cost"]
        )
    )

    if not base_is_valid:
        reason = base[
            "compliance"
        ]["summary"].get(
            "failure_reasons",
            "",
        )

        raise RuntimeError(
            "The common base-case solve failed or was "
            "physically noncompliant; training cannot "
            "define its reward reference. "
            f"Reason: {reason}"
        )

    base_economic_cost = float(
        base["economic_cost"]
    )

    env.set_reference_economic_cost(
        base_economic_cost
    )

    network = ADNDeepNet()

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=1.0e-4,
        weight_decay=1.0e-4,
    )

    replay = ReplayBuffer(
        capacity=10_000
    )

    rng = np.random.default_rng(seed)

    valid_episodes = 0
    invalid_episodes = 0
    optimizer_updates = 0
    final_checkpoint = None

    with log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=LOG_FIELDS,
        )

        writer.writeheader()
        log_handle.flush()

        for episode in range(
            1,
            episodes + 1,
        ):
            episode_start = perf_counter()

            state = env.reset()
            trajectory = []
            stopped = False

            for move in range(max_moves):
                search = MCTS(
                    neural_net=network,
                    terminal_evaluator=(
                        env.evaluate_terminal_value
                    ),
                    num_simulations=simulations,
                    rng=rng,
                    exact_terminal_evaluation=False,
                    stop_prior_floor=(
                        stop_prior_floor
                    ),
                )

                temperature = (
                    1.0
                    if move < 10
                    else 0.1
                )

                action, policy = search.search(
                    state,
                    temperature=temperature,
                    add_noise=True,
                )

                trajectory.append(
                    (
                        search.state_to_tensor(
                            state
                        ),
                        policy,
                    )
                )

                if action[0] == "stop":
                    stopped = True
                    break

                if not state.add_device(
                    *action
                ):
                    break

            # This is the only exact physical solve associated
            # with the current self-play episode.
            terminal = (
                env.evaluate_terminal_state(
                    state,
                    force_solve=True,
                )
            )

            terminal_value = float(
                terminal["terminal_value"]
            )

            sample_used = bool(
                terminal["solver_ok"]
                and np.isfinite(
                    terminal[
                        "economic_cost"
                    ]
                )
                and np.isfinite(
                    terminal_value
                )
            )

            losses = None

            if sample_used:
                perf = performance_target(
                    terminal
                )

                for (
                    encoded_state,
                    policy,
                ) in trajectory:
                    replay.push(
                        encoded_state,
                        policy,
                        terminal_value,
                        perf,
                    )

                valid_episodes += 1

                if len(replay) >= batch_size:
                    for _ in range(
                        updates_per_episode
                    ):
                        batch = replay.sample(
                            batch_size
                        )

                        losses = (
                            network.train_step(
                                optimizer,
                                *batch,
                            )
                        )

                        optimizer_updates += 1
            else:
                invalid_episodes += 1

            summary = terminal[
                "compliance"
            ]["summary"]

            economic_cost = float(
                terminal["economic_cost"]
            )

            cost_change_percent = (
                (
                    economic_cost
                    - base_economic_cost
                )
                / base_economic_cost
                * 100.0
                if np.isfinite(
                    economic_cost
                )
                else float("inf")
            )

            episode_elapsed = (
                perf_counter()
                - episode_start
            )

            row = {
                "episode": episode,
                "seed": seed,
                "moves": len(trajectory),
                "selected_stop": stopped,
                "terminal_value": (
                    terminal_value
                ),
                "solver_ok": terminal[
                    "solver_ok"
                ],
                "compliant": terminal[
                    "is_compliant"
                ],
                "sample_used": sample_used,
                "economic_cost": (
                    economic_cost
                ),
                "cost_change_percent_vs_base": (
                    cost_change_percent
                ),
                "objective_cost": terminal.get(
                    "objective_cost"
                ),
                "penalty_cost": terminal.get(
                    "penalty_cost"
                ),
                "placement": repr(
                    terminal.get(
                        "placement",
                        {},
                    )
                ),
                "placement_ieee": repr(
                    terminal.get(
                        "placement_ieee",
                        {},
                    )
                ),
                "termination_condition": (
                    terminal.get(
                        "termination_condition"
                    )
                ),
                "failure_reasons": (
                    summary.get(
                        "failure_reasons",
                        "",
                    )
                ),
                "max_cone_abs_gap": (
                    summary.get(
                        "max_cone_abs_gap"
                    )
                ),
                "nonexact_cone_points": (
                    summary.get(
                        "nonexact_cone_points"
                    )
                ),
                "min_voltage_pu": (
                    summary.get(
                        "min_voltage_pu"
                    )
                ),
                "res_curtailment_mwh_representative": (
                    summary.get(
                        "res_curtailment_mwh_representative"
                    )
                ),
                "load_shedding_mwh_representative": (
                    summary.get(
                        "load_shedding_mwh_representative"
                    )
                ),
                "valid_episodes_cumulative": (
                    valid_episodes
                ),
                "invalid_episodes_cumulative": (
                    invalid_episodes
                ),
                "replay_samples": len(
                    replay
                ),
                "optimizer_updates_cumulative": (
                    optimizer_updates
                ),
                "physics_solves_cumulative": (
                    env.physics_solve_count
                ),
                "episode_elapsed_seconds": (
                    episode_elapsed
                ),
                "loss_total": (
                    None
                    if losses is None
                    else losses["total"]
                ),
            }

            writer.writerow(row)
            log_handle.flush()

            print(
                (
                    f"Episode {episode}/{episodes} | "
                    f"solver_ok={terminal['solver_ok']} | "
                    f"compliant={terminal['is_compliant']} | "
                    f"sample_used={sample_used} | "
                    f"moves={len(trajectory)} | "
                    f"cost={economic_cost:.6f} | "
                    f"elapsed={episode_elapsed:.1f}s"
                ),
                flush=True,
            )

            should_save_periodic = bool(
                checkpoint_interval > 0
                and episode
                % checkpoint_interval
                == 0
                and valid_episodes > 0
                and len(replay) > 0
            )

            if should_save_periodic:
                periodic_path = (
                    output
                    / (
                        f"adn_{ARCHITECTURE_VERSION}"
                        f"_seed{seed}_{stamp}"
                        f"_episode{episode:04d}.pth"
                    )
                )

                payload = (
                    build_checkpoint_payload(
                        network=network,
                        optimizer=optimizer,
                        seed=seed,
                        episode=episode,
                        configured_episodes=(
                            episodes
                        ),
                        valid_episodes=(
                            valid_episodes
                        ),
                        invalid_episodes=(
                            invalid_episodes
                        ),
                        replay_samples=len(
                            replay
                        ),
                        optimizer_updates=(
                            optimizer_updates
                        ),
                        base_economic_cost=(
                            base_economic_cost
                        ),
                        simulations=(
                            simulations
                        ),
                        max_moves=max_moves,
                        batch_size=batch_size,
                        updates_per_episode=(
                            updates_per_episode
                        ),
                        stop_prior_floor=(
                            stop_prior_floor
                        ),
                    )
                )

                save_checkpoint_atomic(
                    periodic_path,
                    payload,
                )

                print(
                    (
                        "Periodic checkpoint: "
                        f"{periodic_path}"
                    ),
                    flush=True,
                )

    if (
        valid_episodes > 0
        and len(replay) > 0
    ):
        final_checkpoint = (
            output
            / (
                f"adn_{ARCHITECTURE_VERSION}"
                f"_seed{seed}_{stamp}"
                "_final.pth"
            )
        )

        final_payload = (
            build_checkpoint_payload(
                network=network,
                optimizer=optimizer,
                seed=seed,
                episode=episodes,
                configured_episodes=episodes,
                valid_episodes=(
                    valid_episodes
                ),
                invalid_episodes=(
                    invalid_episodes
                ),
                replay_samples=len(replay),
                optimizer_updates=(
                    optimizer_updates
                ),
                base_economic_cost=(
                    base_economic_cost
                ),
                simulations=simulations,
                max_moves=max_moves,
                batch_size=batch_size,
                updates_per_episode=(
                    updates_per_episode
                ),
                stop_prior_floor=(
                    stop_prior_floor
                ),
            )
        )

        save_checkpoint_atomic(
            final_checkpoint,
            final_payload,
        )

    return final_checkpoint, log_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-moves",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--updates-per-episode",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output-dir",
        default="models",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--stop-prior-floor",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path, log_path = train_from_scratch(
        episodes=args.episodes,
        simulations=args.simulations,
        max_moves=args.max_moves,
        seed=args.seed,
        batch_size=args.batch_size,
        updates_per_episode=(
            args.updates_per_episode
        ),
        output_dir=args.output_dir,
        checkpoint_interval=(
            args.checkpoint_interval
        ),
        stop_prior_floor=(
            args.stop_prior_floor
        ),
    )

    if model_path is None:
        print(
            "No checkpoint was created because no valid "
            "training sample was produced."
        )
    else:
        print(
            f"Final checkpoint: {model_path}"
        )

    print(f"Training log: {log_path}")