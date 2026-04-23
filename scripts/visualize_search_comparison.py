#!/usr/bin/env python3
"""Visualize why Breadth First Search and A Star Search explore differently."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_and_visualize as rv


@dataclass
class GroundedOperator:
    action: rv.GroundedAction
    preconditions: List[rv.Condition]
    effects: List[rv.Condition]


@dataclass
class SearchNode:
    state: frozenset[str]
    parent: int
    action: str
    depth: int
    h: int
    sequence: int
    expanded_order: int | None = None


@dataclass
class SearchResult:
    method: str
    solved: bool
    nodes: List[SearchNode]
    goal_index: int | None
    expanded: int
    generated: int
    elapsed_ms: float
    records: List[Dict[str, int]]
    solution_path: List[int]
    solution_actions: List[str]


def state_key(state: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(state))


def generate_grounded_operators(env: rv.Environment) -> List[GroundedOperator]:
    operators: List[GroundedOperator] = []
    for schema in env.actions:
        for values in product(env.symbols, repeat=len(schema.params)):
            binding = dict(zip(schema.params, values))
            preconditions = [rv.bind_condition(cond, binding) for cond in schema.preconditions]
            effects = [rv.bind_condition(effect, binding) for effect in schema.effects]
            operators.append(GroundedOperator(rv.GroundedAction(schema.name, tuple(values)), preconditions, effects))
    return operators


def condition_holds(state: frozenset[str], condition: rv.Condition) -> bool:
    present = condition.fact() in state
    return present if condition.truth else not present


def applicable(state: frozenset[str], op: GroundedOperator) -> bool:
    return all(condition_holds(state, condition) for condition in op.preconditions)


def apply_operator(state: frozenset[str], op: GroundedOperator) -> frozenset[str]:
    next_state = set(state)
    for effect in op.effects:
        if not effect.truth:
            next_state.discard(effect.fact())
    for effect in op.effects:
        if effect.truth:
            next_state.add(effect.fact())
    return frozenset(next_state)


def relaxed_plan_heuristic(state: frozenset[str], env: rv.Environment, operators: Sequence[GroundedOperator]) -> int:
    if goals_satisfied(state, env):
        return 0

    fact_layers: List[set[str]] = [set(state)]
    action_layers: List[List[int]] = []

    while not env.goals <= fact_layers[-1]:
        current = fact_layers[-1]
        applicable_actions: List[int] = []
        next_facts = set(current)

        for index, op in enumerate(operators):
            if all((not pre.truth) or pre.fact() in current for pre in op.preconditions):
                applicable_actions.append(index)
                for effect in op.effects:
                    if effect.truth:
                        next_facts.add(effect.fact())

        if next_facts == current:
            return 1_000_000

        action_layers.append(applicable_actions)
        fact_layers.append(next_facts)

    needed: List[set[str]] = [set() for _ in fact_layers]
    needed[-1] = set(env.goals)
    selected: set[str] = set()

    for layer in range(len(fact_layers) - 1, 0, -1):
        for goal in sorted(needed[layer]):
            if goal in fact_layers[layer - 1]:
                continue

            achievers = [
                operators[index]
                for index in action_layers[layer - 1]
                if any(effect.truth and effect.fact() == goal for effect in operators[index].effects)
            ]
            if not achievers:
                return 1_000_000

            chosen = min(
                achievers,
                key=lambda op: (
                    sum(1 for pre in op.preconditions if pre.truth and pre.fact() not in fact_layers[layer - 1]),
                    len(op.preconditions),
                    op.action.text(),
                ),
            )
            selected.add(chosen.action.text())
            for precondition in chosen.preconditions:
                if precondition.truth:
                    needed[layer - 1].add(precondition.fact())

    return len(selected)


def heuristic(state: frozenset[str], env: rv.Environment, operators: Sequence[GroundedOperator]) -> int:
    return relaxed_plan_heuristic(state, env, operators)


def goal_count(state: frozenset[str], env: rv.Environment) -> int:
    return sum(1 for goal in env.goals if goal in state)


def goals_satisfied(state: frozenset[str], env: rv.Environment) -> bool:
    return all(goal in state for goal in env.goals)


def reconstruct(nodes: Sequence[SearchNode], goal_index: int | None) -> Tuple[List[int], List[str]]:
    if goal_index is None:
        return [], []
    path: List[int] = []
    actions: List[str] = []
    index = goal_index
    while index >= 0:
        path.append(index)
        if nodes[index].action:
            actions.append(nodes[index].action)
        index = nodes[index].parent
    path.reverse()
    actions.reverse()
    return path, actions


def make_record(expanded: int, generated: int, frontier: int, nodes: Sequence[SearchNode], env: rv.Environment) -> Dict[str, int]:
    best_goal_count = max((goal_count(node.state, env) for node in nodes), default=0)
    min_h = min((node.h for node in nodes), default=len(env.goals))
    return {
        "expanded": expanded,
        "generated": generated,
        "frontier": frontier,
        "best_goal_count": best_goal_count,
        "min_h": min_h,
    }


def run_bfs(env: rv.Environment, max_expansions: int) -> SearchResult:
    start_time = time.perf_counter()
    operators = generate_grounded_operators(env)
    nodes = [SearchNode(frozenset(env.initial), -1, "", 0, heuristic(frozenset(env.initial), env, operators), 0)]
    open_queue: deque[int] = deque([0])
    visited = {state_key(nodes[0].state)}
    records: List[Dict[str, int]] = []
    expanded = 0
    goal_index = 0 if goals_satisfied(nodes[0].state, env) else None

    while open_queue and goal_index is None and expanded < max_expansions:
        node_index = open_queue.popleft()
        nodes[node_index].expanded_order = expanded + 1
        expanded += 1
        current = nodes[node_index]
        for op in operators:
            if not applicable(current.state, op):
                continue
            next_state = apply_operator(current.state, op)
            key = state_key(next_state)
            if key in visited:
                continue
            visited.add(key)
            child_index = len(nodes)
            nodes.append(
                SearchNode(
                    next_state,
                    node_index,
                    op.action.text(),
                    current.depth + 1,
                    heuristic(next_state, env, operators),
                    child_index,
                )
            )
            open_queue.append(child_index)
            if goals_satisfied(next_state, env):
                goal_index = child_index
                break
        records.append(make_record(expanded, len(nodes) - 1, len(open_queue), nodes, env))

    path, actions = reconstruct(nodes, goal_index)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult("Breadth First Search", goal_index is not None, nodes, goal_index, expanded, len(nodes) - 1, elapsed_ms, records, path, actions)


def run_astar(env: rv.Environment, max_expansions: int) -> SearchResult:
    start_time = time.perf_counter()
    operators = generate_grounded_operators(env)
    start_state = frozenset(env.initial)
    nodes = [SearchNode(start_state, -1, "", 0, heuristic(start_state, env, operators), 0)]
    heap: List[Tuple[int, int, int, int]] = [(nodes[0].h, 0, 0, 0)]
    best_g = {state_key(start_state): 0}
    records: List[Dict[str, int]] = []
    expanded = 0
    sequence = 1
    goal_index: int | None = None

    while heap and expanded < max_expansions:
        _, neg_g, _, node_index = heapq.heappop(heap)
        current = nodes[node_index]
        current_g = -neg_g
        if best_g.get(state_key(current.state), current_g) != current_g:
            continue
        if goals_satisfied(current.state, env):
            goal_index = node_index
            break
        current.expanded_order = expanded + 1
        expanded += 1
        for op in operators:
            if not applicable(current.state, op):
                continue
            next_state = apply_operator(current.state, op)
            next_key = state_key(next_state)
            next_g = current_g + 1
            if best_g.get(next_key, 1_000_000_000) <= next_g:
                continue
            best_g[next_key] = next_g
            child_index = len(nodes)
            h_value = heuristic(next_state, env, operators)
            nodes.append(SearchNode(next_state, node_index, op.action.text(), current.depth + 1, h_value, child_index))
            heapq.heappush(heap, (next_g + h_value, -next_g, sequence, child_index))
            sequence += 1
        records.append(make_record(expanded, len(nodes) - 1, len(heap), nodes, env))

    path, actions = reconstruct(nodes, goal_index)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult("A Star Search [relaxed-plan delete relaxation]", goal_index is not None, nodes, goal_index, expanded, len(nodes) - 1, elapsed_ms, records, path, actions)


def select_nodes(result: SearchResult, max_tree_nodes: int) -> List[int]:
    selected = set(result.solution_path)
    for node in result.nodes:
        if len(selected) >= max_tree_nodes:
            break
        selected.add(node.sequence)
    return sorted(selected, key=lambda idx: (result.nodes[idx].depth, result.nodes[idx].sequence))


def draw_search_tree(ax, result: SearchResult, max_tree_nodes: int) -> None:
    selected = set(select_nodes(result, max_tree_nodes))
    by_depth: Dict[int, List[int]] = defaultdict(list)
    for idx in sorted(selected, key=lambda i: result.nodes[i].sequence):
        by_depth[result.nodes[idx].depth].append(idx)

    positions: Dict[int, Tuple[float, float]] = {}
    for depth, indices in by_depth.items():
        count = len(indices)
        for rank, idx in enumerate(indices):
            positions[idx] = (depth, (count - 1) / 2.0 - rank)

    solution_edges = set(zip(result.solution_path[:-1], result.solution_path[1:]))
    for idx in selected:
        node = result.nodes[idx]
        if node.parent in selected:
            x0, y0 = positions[node.parent]
            x1, y1 = positions[idx]
            is_solution_edge = (node.parent, idx) in solution_edges
            ax.plot([x0, x1], [y0, y1], color="#f97316" if is_solution_edge else "#94a3b8", linewidth=2.2 if is_solution_edge else 0.65, alpha=0.95 if is_solution_edge else 0.25)

    xs = [positions[idx][0] for idx in selected]
    ys = [positions[idx][1] for idx in selected]
    hs = [result.nodes[idx].h for idx in selected]
    solution_set = set(result.solution_path)
    sizes = [95 if idx in solution_set else 28 for idx in selected]
    edge_colors = ["#f97316" if idx in solution_set else "#334155" for idx in selected]
    scatter = ax.scatter(xs, ys, c=hs, cmap="viridis_r", s=sizes, edgecolors=edge_colors, linewidths=0.8, zorder=3)

    for idx in result.solution_path:
        if idx in positions and result.nodes[idx].action:
            x, y = positions[idx]
            ax.text(x + 0.05, y + 0.25, result.nodes[idx].action.split("(")[0], fontsize=7, color="#9a3412")

    ax.set_title(
        f"{result.method}\nexpanded={result.expanded}, generated={result.generated}, plan={len(result.solution_actions)}",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    ax.set_xlabel("search depth")
    ax.set_ylabel("generated states at depth")
    ax.grid(True, alpha=0.18)
    ax.text(
        0.01,
        0.02,
        f"Nodes shown: first {min(max_tree_nodes, len(result.nodes))} generated plus solution path\nColor = unsatisfied goal facts, orange = final plan",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        color="#334155",
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.90),
    )
    return scatter


def draw_effort_curves(ax, bfs: SearchResult, astar: SearchResult) -> None:
    for result, color in [(bfs, "#2563eb"), (astar, "#dc2626")]:
        xs = [row["expanded"] for row in result.records]
        generated = [row["generated"] for row in result.records]
        frontier = [row["frontier"] for row in result.records]
        ax.plot(xs, generated, color=color, linewidth=2.2, label=f"{result.method} generated")
        ax.plot(xs, frontier, color=color, linestyle="--", linewidth=1.6, label=f"{result.method} frontier")
    ax.set_title("Search Work Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("expanded states")
    ax.set_ylabel("state count")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def draw_goal_curves(ax, env: rv.Environment, bfs: SearchResult, astar: SearchResult) -> None:
    for result, color in [(bfs, "#2563eb"), (astar, "#dc2626")]:
        xs = [row["expanded"] for row in result.records]
        best = [row["best_goal_count"] for row in result.records]
        min_h = [row["min_h"] for row in result.records]
        ax.plot(xs, best, color=color, linewidth=2.2, label=f"{result.method} best goals")
        ax.plot(xs, [len(env.goals) - value for value in min_h], color=color, linestyle="--", linewidth=1.6, label=f"{result.method} best h")
    ax.set_title("How Close the Search Gets to the Goal", fontsize=12, fontweight="bold")
    ax.set_xlabel("expanded states")
    ax.set_ylabel("satisfied goal facts")
    ax.set_ylim(-0.1, len(env.goals) + 0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def draw_summary(ax, env: rv.Environment, bfs: SearchResult, astar: SearchResult) -> None:
    ax.axis("off")
    lines = [
        f"Environment: {env.name}",
        f"Goal facts: {len(env.goals)}",
        "",
        "Metric                         BFS        A Star",
        f"valid solution                  {str(bfs.solved):<10} {str(astar.solved):<10}",
        f"plan length                     {len(bfs.solution_actions):<10} {len(astar.solution_actions):<10}",
        f"expanded states                 {bfs.expanded:<10} {astar.expanded:<10}",
        f"generated states                {bfs.generated:<10} {astar.generated:<10}",
        f"python search time ms           {bfs.elapsed_ms:>9.2f} {astar.elapsed_ms:>10.2f}",
        "",
        "Why this differs:",
        "BFS expands by depth without estimating remaining work.",
        "A Star uses a delete-relaxed relaxed-plan estimate. It builds a monotone",
        "planning graph from each state, extracts a relaxed plan, and uses the",
        "relaxed action count as h(s). This is stronger than raw goal counting.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=10, family="DejaVu Sans Mono", va="top", color="#111827")


def record_at(result: SearchResult, expanded_target: int) -> Dict[str, int]:
    if not result.records:
        return {"expanded": 0, "generated": 0, "frontier": 0, "best_goal_count": 0, "min_h": 0}
    current = result.records[0]
    for row in result.records:
        if row["expanded"] > expanded_target:
            break
        current = row
    return current


def draw_progress_bar(ax, label: str, value: int, maximum: int, color: str, y: float) -> None:
    ax.text(0.02, y + 0.045, label, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#0f172a", va="center")
    ax.add_patch(plt.Rectangle((0.24, y), 0.70, 0.07, transform=ax.transAxes, color="#e2e8f0", ec="#cbd5e1"))
    width = 0.70 * (value / max(1, maximum))
    ax.add_patch(plt.Rectangle((0.24, y), width, 0.07, transform=ax.transAxes, color=color, ec=color))
    ax.text(0.955, y + 0.035, f"{value}", transform=ax.transAxes, fontsize=10, color="#111827", va="center", ha="left")


def build_search_animation(env: rv.Environment, bfs: SearchResult, astar: SearchResult, out_dir: Path, env_label: str) -> Path:
    frames_dir = out_dir / f"_{env_label}_search_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    gif_path = out_dir / f"{env_label}_bfs_vs_astar_search.gif"
    max_expanded = max(bfs.expanded, astar.expanded, 1)
    frame_targets = sorted(set(int(max_expanded * t / 18) for t in range(19)))
    image_paths: List[Path] = []

    bfs_records = bfs.records or [{"expanded": 0, "generated": 0, "frontier": 0, "best_goal_count": 0, "min_h": len(env.goals)}]
    astar_records = astar.records or [{"expanded": 0, "generated": 0, "frontier": 0, "best_goal_count": 0, "min_h": len(env.goals)}]

    for frame_index, target in enumerate(frame_targets):
        bfs_row = record_at(bfs, target)
        astar_row = record_at(astar, target)

        fig = plt.figure(figsize=(13.2, 7.4), facecolor="#f8fafc")
        fig.suptitle(f"{env_label} | Search Expansion Animation | expanded target {target}", fontsize=15, fontweight="bold", color="#0f172a")
        grid = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.30, wspace=0.24)

        ax_bars = fig.add_subplot(grid[0, :])
        ax_bars.axis("off")
        draw_progress_bar(ax_bars, "BFS generated states", bfs_row["generated"], max(bfs.generated, astar.generated, 1), "#2563eb", 0.74)
        draw_progress_bar(ax_bars, "A* generated states", astar_row["generated"], max(bfs.generated, astar.generated, 1), "#dc2626", 0.58)
        draw_progress_bar(ax_bars, "BFS frontier", bfs_row["frontier"], max(max(r["frontier"] for r in bfs_records), max(r["frontier"] for r in astar_records), 1), "#60a5fa", 0.36)
        draw_progress_bar(ax_bars, "A* frontier", astar_row["frontier"], max(max(r["frontier"] for r in bfs_records), max(r["frontier"] for r in astar_records), 1), "#f87171", 0.20)
        ax_bars.text(
            0.02,
            0.02,
            f"BFS: expanded {bfs_row['expanded']} / {bfs.expanded}, best goal facts {bfs_row['best_goal_count']} / {len(env.goals)}    "
            f"A*: expanded {astar_row['expanded']} / {astar.expanded}, best goal facts {astar_row['best_goal_count']} / {len(env.goals)}",
            transform=ax_bars.transAxes,
            fontsize=10,
            color="#334155",
        )

        ax_work = fig.add_subplot(grid[1, 0])
        for result, color, row in [(bfs, "#2563eb", bfs_row), (astar, "#dc2626", astar_row)]:
            partial = [r for r in result.records if r["expanded"] <= row["expanded"]]
            if partial:
                ax_work.plot([r["expanded"] for r in partial], [r["generated"] for r in partial], color=color, linewidth=2.2, label=f"{result.method} generated")
                ax_work.plot([r["expanded"] for r in partial], [r["frontier"] for r in partial], color=color, linestyle="--", linewidth=1.5, label=f"{result.method} frontier")
        ax_work.set_xlim(0, max_expanded * 1.02)
        ax_work.set_ylim(0, max(bfs.generated, astar.generated) * 1.08)
        ax_work.set_title("Generated States and Frontier Growth", fontsize=12, fontweight="bold")
        ax_work.set_xlabel("expanded states")
        ax_work.set_ylabel("states")
        ax_work.grid(True, alpha=0.25)
        ax_work.legend(fontsize=8)

        ax_goal = fig.add_subplot(grid[1, 1])
        for result, color, row in [(bfs, "#2563eb", bfs_row), (astar, "#dc2626", astar_row)]:
            partial = [r for r in result.records if r["expanded"] <= row["expanded"]]
            if partial:
                ax_goal.plot([r["expanded"] for r in partial], [r["best_goal_count"] for r in partial], color=color, linewidth=2.2, label=f"{result.method} best goals")
        ax_goal.set_xlim(0, max_expanded * 1.02)
        ax_goal.set_ylim(-0.1, len(env.goals) + 0.4)
        ax_goal.set_title("Best Goal Satisfaction Found So Far", fontsize=12, fontweight="bold")
        ax_goal.set_xlabel("expanded states")
        ax_goal.set_ylabel("satisfied goal facts")
        ax_goal.grid(True, alpha=0.25)
        ax_goal.legend(fontsize=8)

        frame_path = frames_dir / f"frame_{frame_index:03d}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(frame_path)

    rv.save_stable_gif(image_paths, gif_path, duration=360)
    return gif_path


def build_visualization(env_name: str, max_expansions: int, max_tree_nodes: int) -> Path:
    env = rv.parse_environment(rv.ENVS_DIR / env_name)
    bfs = run_bfs(env, max_expansions)
    astar = run_astar(env, max_expansions)
    out_dir = rv.VIS_ROOT / "search_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_label = env_name.replace(".txt", "")
    png_path = out_dir / f"{env_label}_bfs_vs_astar_search.png"
    gif_path = out_dir / f"{env_label}_bfs_vs_astar_search.gif"
    json_path = out_dir / f"{env_label}_bfs_vs_astar_search.json"

    fig = plt.figure(figsize=(18, 12), facecolor="#f8fafc")
    fig.suptitle(f"{env_label} | Breadth First Search versus A Star Search with Delete Relaxation", fontsize=17, fontweight="bold", color="#0f172a")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.74, 0.62], hspace=0.36, wspace=0.20)
    draw_search_tree(fig.add_subplot(grid[0, 0]), bfs, max_tree_nodes)
    draw_search_tree(fig.add_subplot(grid[0, 1]), astar, max_tree_nodes)
    draw_effort_curves(fig.add_subplot(grid[1, 0]), bfs, astar)
    draw_goal_curves(fig.add_subplot(grid[1, 1]), env, bfs, astar)
    draw_summary(fig.add_subplot(grid[2, :]), env, bfs, astar)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gif_path = build_search_animation(env, bfs, astar, out_dir, env_label)

    json_path.write_text(
        json.dumps(
            {
                "environment": env_name,
                "bfs": {
                    "solved": bfs.solved,
                    "plan_length": len(bfs.solution_actions),
                    "expanded": bfs.expanded,
                    "generated": bfs.generated,
                    "elapsed_ms": bfs.elapsed_ms,
                    "solution_actions": bfs.solution_actions,
                },
                "astar": {
                    "solved": astar.solved,
                    "plan_length": len(astar.solution_actions),
                    "expanded": astar.expanded,
                    "generated": astar.generated,
                    "elapsed_ms": astar.elapsed_ms,
                    "solution_actions": astar.solution_actions,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{env_name}: BFS expanded={bfs.expanded} generated={bfs.generated}; A* expanded={astar.expanded} generated={astar.generated}")
    print(png_path)
    print(gif_path)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw BFS versus A* search effort panels.")
    parser.add_argument("--env", action="append", default=[], help="Environment file to visualize. Can be repeated.")
    parser.add_argument("--max-expansions", type=int, default=20000)
    parser.add_argument("--max-tree-nodes", type=int, default=180)
    args = parser.parse_args()

    envs = args.env or ["BlocksTwoStack5.txt", "BlocksTriangleBridge.txt"]
    for env_name in envs:
        build_visualization(env_name, args.max_expansions, args.max_tree_nodes)


if __name__ == "__main__":
    main()
