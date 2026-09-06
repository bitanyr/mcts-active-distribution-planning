"""Honest hard evaluation of a newly trained corrected checkpoint."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from core.network import ADNDeepNet
from core.planning import hard_finalize, mcts_plan
from core.reporting import flatten_summary, write_solution_snapshot
from env.aps_env import ActivePlanningEnv
from env.state import NetworkState
from optimization.compliance import export_compliance_to_excel
from optimization.solver import evaluate_placement
from self_play import ARCHITECTURE_VERSION


def load_corrected_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    version = checkpoint.get("architecture_version")
    if version != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint {path} is {version!r}; expected {ARCHITECTURE_VERSION!r}. "
            "Do not evaluate the old episode-200 model. Run self_play.py from scratch."
        )
    network = ADNDeepNet()
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()
    return network, checkpoint


def make_compliance_plot(results, output_path):
    labels = list(results)
    cone = [results[name]["compliance"]["summary"].get("max_cone_abs_gap", float("nan")) for name in labels]
    slack = [results[name]["compliance"]["summary"].get("max_slack", float("nan")) for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    axes[0].bar(labels, cone, color="#2F6B8A")
    axes[0].axhline(1.0e-6, color="#B23A48", linestyle="--", label="abs tolerance")
    axes[0].set_yscale("log")
    axes[0].set_title("Maximum absolute cone gap")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, slack, color="#6C8E4E")
    axes[1].axhline(1.0e-6, color="#B23A48", linestyle="--", label="slack tolerance")
    axes[1].set_yscale("log")
    axes[1].set_title("Maximum artificial slack")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)

    plt.close(fig)


def evaluate(checkpoint_path, output_dir="results", budget=500, simulations=500, max_moves=32, seed=0):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = ActivePlanningEnv()

    base_soft = evaluate_placement(env.model, {}, hard_verify=False)
    if not base_soft["solver_ok"]:
        raise RuntimeError("Base case did not solve, so no common reward reference exists.")
    env.set_reference_economic_cost(base_soft["economic_cost"])
    network, metadata = load_corrected_checkpoint(checkpoint_path)

    workbook = output / "Grid_Simulation_Results_corrected.xlsx"
    summaries = []
    results = {}
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        base_state = NetworkState()
        base_hard = hard_finalize(env, base_state)
        results["Base"] = base_hard
        summaries.append(flatten_summary("Base", base_hard))
        if base_hard["solver_ok"]:
            write_solution_snapshot(writer, env.model, "Base")
        export_compliance_to_excel(base_hard["compliance"], "Base", writer)

        planned_state, search_record, used, _ = mcts_plan(
            env,
            network,
            evaluation_budget=budget,
            max_moves=max_moves,
            simulations_per_move=simulations,
            seed=seed,
            add_noise=False,
        )
        trained_hard = hard_finalize(env, planned_state)
        trained_hard["search_physics_solves"] = used
        trained_hard["search_soft_record"] = search_record
        results["MCTS"] = trained_hard
        summaries.append(flatten_summary("MCTS", trained_hard))
        if trained_hard["solver_ok"]:
            write_solution_snapshot(writer, env.model, "MCTS")
        export_compliance_to_excel(trained_hard["compliance"], "MCTS", writer)

        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(
            [{"checkpoint": str(checkpoint_path), **metadata}]
        ).drop(columns=["state_dict"]).to_excel(writer, sheet_name="Run_Metadata", index=False)

    figure = output / "Compliance_Audit_corrected.png"
    make_compliance_plot(results, figure)
    return results, workbook, figure



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="A corrected-v3 checkpoint produced by self_play.py")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--max-moves", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records, workbook_path, figure_path = evaluate(
        args.checkpoint,
        output_dir=args.output_dir,
        budget=args.budget,
        simulations=args.simulations,
        max_moves=args.max_moves,
        seed=args.seed,
    )
    print("Method       Solver  Compliance  Hard verified  Annual economic cost")
    for name, record in records.items():
        cost_text = (
            f"{record['economic_cost']:.2f}"
            if record["hard_verified"]
            else "N/A (not hard-verified)"
        )
        print(
            f"{name:<12} {str(record['solver_ok']):<7} {str(record['is_compliant']):<11} "
            f"{str(record['hard_verified']):<13} {cost_text}"
        )
    print(f"Workbook: {workbook_path}")
    print(f"Compliance figure: {figure_path}")