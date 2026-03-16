"""Headless LLAMOSC simulation runner.

Runs the LLAMOSC simulation without GUI (PyQt5) or Docker dependencies.
Supports both LLM mode (Ollama) and test mode (mock data).

Usage:
    uv run llamosc-headless --contributors 5 --maintainers 3 --issues 5 --algorithm a
    uv run llamosc-headless --test  # fast mock mode, no LLM needed
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from llamosc.agents.contributor import ContributorAgent
from llamosc.agents.maintainer import MaintainerAgent
from llamosc.agents.issue_creator import IssueCreatorAgent
from llamosc.simulation.issue import Issue
from llamosc.simulation.sim import Simulation
from llamosc.utils import log_and_print, log, repo_commit_current_changes


def create_test_issues(issues_folder: str, n_issues: int) -> list[Issue]:
    """Create synthetic test issues for mock mode."""
    os.makedirs(issues_folder, exist_ok=True)
    issues = []
    templates = [
        "Add trigonometric functions (sin, cos, tan) to the calculator",
        "Implement memory storage (M+, M-, MR, MC) functionality",
        "Add support for complex number arithmetic",
        "Implement unit conversion feature (length, weight, temperature)",
        "Add expression history with undo/redo support",
        "Implement matrix operations (add, multiply, determinant)",
        "Add graphing capability for simple functions",
        "Implement statistical functions (mean, median, std dev)",
        "Add support for different number bases (hex, octal, binary)",
        "Implement a plugin system for custom operations",
    ]
    for i in range(1, n_issues + 1):
        filepath = os.path.join(issues_folder, f"task_{i}.md")
        with open(filepath, "w") as f:
            template = templates[(i - 1) % len(templates)]
            f.write(
                f"# Issue {i}: {template}\n\n"
                f"## Description\n{template}.\n\n"
                f"## Difficulty\n{(i % 5) + 1}\n"
            )
        issues.append(Issue(id=i, difficulty=(i % 4) + 1, filepath=filepath))
    return issues


def read_issues(issues_folder: str) -> list[Issue]:
    """Read issue files from a folder and return Issue objects."""
    issues = []
    if not os.path.exists(issues_folder):
        return issues
    for filename in sorted(os.listdir(issues_folder)):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(issues_folder, filename)
        try:
            issue_id = int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        issue = Issue(issue_id, (issue_id + 1) % 5, file_path)
        issues.append(issue)
    return issues


def run_simulation(
    n_contributors: int,
    n_maintainers: int,
    n_issues: int,
    algorithm: str,
    test_mode: bool,
    project_dir: str,
    output_dir: str,
    seed: int | None = None,
) -> dict:
    """Run the LLAMOSC simulation and return results."""
    if seed is not None:
        random.seed(seed)

    issues_parent = os.path.join(project_dir, "issues")
    issues_folder = os.path.join(issues_parent, "pending")
    pull_requests_dir = os.path.join(project_dir, "pull_requests")
    os.makedirs(issues_folder, exist_ok=True)
    os.makedirs(pull_requests_dir, exist_ok=True)

    # Load or create issues
    if test_mode:
        issues = create_test_issues(issues_folder, n_issues)
    else:
        issues = read_issues(issues_folder)
        if len(issues) < n_issues:
            issue_creator = IssueCreatorAgent(name="Issue Creator")
            existing_code = ""
            for root, _, files in os.walk(project_dir):
                for file in files:
                    if file.endswith(".py"):
                        with open(os.path.join(root, file), "r") as code_file:
                            existing_code += code_file.read() + "\n"
            for _ in range(len(issues), n_issues):
                issue = issue_creator.create_issue(issues, existing_code, issues_folder)
                issues.append(issue)

    # Create agents
    contributors = [
        ContributorAgent(i, random.randint(1, 4), f"Contributor_{i}", testing=test_mode)
        for i in range(n_contributors)
    ]
    maintainers = [
        MaintainerAgent(i, random.randint(4, 5), f"Maintainer_{i}")
        for i in range(n_maintainers)
    ]

    sim = Simulation(contributors)
    log_and_print(
        f"Starting simulation: {len(issues)} issues, {n_contributors} contributors, "
        f"{n_maintainers} maintainers, algorithm={algorithm}, test_mode={test_mode}"
    )

    # Tracking
    experience_history = {c.name: [c.experience] for c in contributors}
    motivation_history = {c.name: [c.motivation_level] for c in contributors}
    code_qal_history = [2.5]
    code_qal_curr_history = [2.5]
    time_history = [0]
    results_per_step = []

    for timestep, issue in enumerate(issues, 1):
        sim.time_step = timestep
        log_and_print(f"\n{'='*60}\nTime Step: {timestep}\n{'='*60}")

        issue_description = open(issue.filepath).read()
        log_and_print(f"Issue #{issue.id} (Difficulty {issue.difficulty}): {issue_description[:100]}...")

        # Select maintainer
        eligible_maintainers = [m for m in maintainers if m.eligible_for_issue(issue)]
        if not eligible_maintainers:
            log_and_print(f"No eligible maintainers for Issue #{issue.id}. Skipping.")
            continue
        selected_maintainer = random.choice(eligible_maintainers)
        selected_maintainer.allot_task(issue)

        # Select contributor
        if test_mode:
            eligible = [c for c in contributors if c.eligible_for_issue(issue)]
            if not eligible:
                log_and_print(f"No eligible contributors for Issue #{issue.id}. Skipping.")
                selected_maintainer.unassign_task()
                continue
            selected_contributor = random.choice(eligible)
            discussion_history = []
        else:
            if algorithm == "d":
                result = sim.select_contributor_decentralized(issue)
            elif algorithm == "c":
                result = sim.select_contributor_collaborative(issue)
            else:
                result = sim.select_contributor_authoritarian(selected_maintainer)

            if result is None:
                log_and_print(f"No eligible contributors for Issue #{issue.id}. Skipping.")
                selected_maintainer.unassign_task()
                continue
            selected_contributor, discussion_history = result

        log_and_print(f"Selected: {selected_contributor.name}")

        # Update motivation for non-selected
        for other in contributors:
            if other.id == selected_contributor.id or not other.eligible_for_issue(issue):
                continue
            if test_mode:
                other.motivation_level = max(0, other.motivation_level - 0.5)
            else:
                other.update_motivation_level(bid_selected=False)

        # Solve issue
        selected_contributor.assign_issue(issue)
        if test_mode:
            task_solved = selected_contributor.solve_issue_without_acr(project_dir, is_test=True)
        else:
            task_solved = selected_contributor.solve_issue_without_acr(project_dir)

        # Review PR
        pr_accepted = False
        code_quality = 0
        if task_solved:
            task_id = issue.id
            pr_dirs = [
                f for f in os.listdir(pull_requests_dir)
                if f.startswith(f"pull_request_{task_id}")
            ]
            if pr_dirs:
                pr_dirs.sort(key=lambda x: int(x.split("_v")[-1]))
                most_recent = pr_dirs[-1]
                pr_dir = os.path.join(pull_requests_dir, most_recent)

                if test_mode:
                    pr_accepted = random.randint(2, 5)
                    code_quality = pr_accepted
                    selected_maintainer.unassign_task()
                else:
                    pr_accepted = selected_maintainer.review_pull_request(pr_dir, project_dir)
                    code_quality = pr_accepted if pr_accepted else 0

                if pr_accepted:
                    selected_contributor.increase_experience(1, issue.difficulty)
                    try:
                        sim.update_code_quality(pr_accepted)
                    except Exception:
                        sim.update_code_quality(random.randint(1, 3))

                    # Move PR to merged
                    merged_dir = os.path.join(pull_requests_dir, "merged")
                    os.makedirs(merged_dir, exist_ok=True)
                    merged_path = os.path.join(merged_dir, most_recent)
                    if os.path.exists(pr_dir):
                        os.rename(pr_dir, merged_path)

                    # Move issue to solved
                    solved_dir = os.path.join(issues_parent, "solved")
                    os.makedirs(solved_dir, exist_ok=True)
                    solved_path = os.path.join(solved_dir, f"task_{task_id}.md")
                    if os.path.exists(issue.filepath):
                        os.rename(issue.filepath, solved_path)

                    sim.issues_solved += 1
                    log_and_print(f"PR MERGED for Issue #{issue.id}, code_quality={code_quality}")
                else:
                    log_and_print(f"PR REJECTED for Issue #{issue.id}")

        # Update motivation
        if test_mode:
            selected_contributor.motivation_level = min(10, selected_contributor.motivation_level + 0.5)
            selected_contributor.motivation_history.append(selected_contributor.motivation_level)
        else:
            selected_contributor.update_motivation_level(
                success=bool(pr_accepted),
                bid_selected=True,
                task_difficulty=issue.difficulty,
                code_quality=code_quality if code_quality else random.randrange(0, 3),
            )

        # Record history
        time_history.append(timestep)
        for c in contributors:
            experience_history[c.name].append(c.experience)
            motivation_history[c.name].append(c.motivation_level)
        code_qal_history.append(sim.avg_code_quality)
        code_qal_curr_history.append(code_quality if code_quality else random.randint(1, 3))

        step_result = {
            "timestep": timestep,
            "issue_id": issue.id,
            "issue_difficulty": issue.difficulty,
            "selected_contributor": selected_contributor.name,
            "pr_accepted": bool(pr_accepted),
            "code_quality": code_quality,
            "avg_code_quality": round(sim.avg_code_quality, 3),
            "issues_solved": sim.issues_solved,
        }
        results_per_step.append(step_result)
        log_and_print(f"Step result: {step_result}")

    # Build summary
    summary = {
        "config": {
            "n_contributors": n_contributors,
            "n_maintainers": n_maintainers,
            "n_issues": n_issues,
            "algorithm": algorithm,
            "test_mode": test_mode,
            "seed": seed,
        },
        "results": results_per_step,
        "final_metrics": {
            "total_issues": len(issues),
            "issues_solved": sim.issues_solved,
            "solve_rate": round(sim.issues_solved / max(len(issues), 1), 3),
            "avg_code_quality": round(sim.avg_code_quality, 3),
            "total_pull_requests": sim.num_pull_requests,
        },
        "agent_final_state": {
            c.name: {
                "experience": round(c.experience, 2),
                "motivation": round(c.motivation_level, 2),
            }
            for c in contributors
        },
        "history": {
            "time": time_history,
            "experience": experience_history,
            "motivation": motivation_history,
            "code_quality_avg": code_qal_history,
            "code_quality_current": code_qal_curr_history,
        },
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "llamosc_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_and_print(f"\nResults saved to {results_path}")

    # Generate plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # Experience plot
    ax = axes[0]
    for c in contributors:
        ax.plot(time_history, experience_history[c.name], label=c.name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Experience")
    ax.set_title("Contributor Experience Over Time")
    ax.legend(loc="upper left", fontsize=8)

    # Code quality plot
    ax = axes[1]
    ax.plot(time_history, code_qal_history, label="Average", color="blue")
    ax.plot(time_history, code_qal_curr_history, label="Current", color="red", alpha=0.6)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Code Quality")
    ax.set_title("Code Quality Over Time")
    ax.legend()

    # Motivation plot
    ax = axes[2]
    for c in contributors:
        ax.plot(time_history, motivation_history[c.name], label=c.name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Motivation")
    ax.set_title("Contributor Motivation Over Time")
    ax.legend(loc="upper left", fontsize=8)

    plot_path = os.path.join(output_dir, "llamosc_metrics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log_and_print(f"Plots saved to {plot_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LLAMOSC Headless Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run llamosc-headless --test                      # Quick test run (no LLM)
  uv run llamosc-headless --test --seed 42            # Reproducible test
  uv run llamosc-headless --contributors 8 --issues 10  # Full LLM run
  uv run llamosc-headless --algorithm d               # Decentralized mode
        """,
    )
    parser.add_argument("--contributors", type=int, default=5, help="Number of contributors (default: 5)")
    parser.add_argument("--maintainers", type=int, default=3, help="Number of maintainers (default: 3)")
    parser.add_argument("--issues", type=int, default=5, help="Number of issues (default: 5)")
    parser.add_argument(
        "--algorithm", type=str, default="a", choices=["a", "d", "c"],
        help="Algorithm: a=authoritarian, d=decentralized, c=collaborative (default: a)",
    )
    parser.add_argument("--test", action="store_true", help="Test mode (mock data, no LLM needed)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Calculator project directory (default: ./calculator_project)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: ./output)",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_dir = args.project_dir or os.path.join(base_dir, "calculator_project")
    output_dir = args.output_dir or os.path.join(base_dir, "output")

    summary = run_simulation(
        n_contributors=args.contributors,
        n_maintainers=args.maintainers,
        n_issues=args.issues,
        algorithm=args.algorithm,
        test_mode=args.test,
        project_dir=project_dir,
        output_dir=output_dir,
        seed=args.seed,
    )

    # Print summary
    fm = summary["final_metrics"]
    print(f"\n{'='*60}")
    print(f"LLAMOSC Simulation Complete")
    print(f"{'='*60}")
    print(f"  Issues solved: {fm['issues_solved']}/{fm['total_issues']} ({fm['solve_rate']*100:.0f}%)")
    print(f"  Avg code quality: {fm['avg_code_quality']:.2f}/5")
    print(f"  Total PRs: {fm['total_pull_requests']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
