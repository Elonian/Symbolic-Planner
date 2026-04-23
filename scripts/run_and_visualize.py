#!/usr/bin/env python3
"""Run and visualize the Symbolic Planner environments.

The assignment planner is a STRIPS state-space planner: each node is a set of
true grounded facts, each edge is an applicable grounded action, and each edge
application adds and deletes facts. This script makes that process inspectable.

Outputs are written under:
  outputs/runs/<environment>/<method>/
  outputs/visualizations/<environment>/<method>/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVS_DIR = PROJECT_ROOT / "envs"
BUILD_DIR = PROJECT_ROOT / "build"
PLANNER_EXE = BUILD_DIR / "planner"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUNS_ROOT = OUTPUT_ROOT / "runs"
VIS_ROOT = OUTPUT_ROOT / "visualizations"

ENVIRONMENTS = [
    "BlocksEasy.txt",
    "Blocks.txt",
    "BlocksTower4.txt",
    "BlocksTwoStack5.txt",
    "BlocksSixPairing.txt",
    "BlocksTriangle.txt",
    "BlocksTriangleBridge.txt",
    "BlocksTriangleTwinTowers.txt",
    "FireExtinguisher.txt",
    "FireExtinguisherReturn.txt",
    "HospitalThreeRobotDelivery.txt",
    "DisasterResponseThreeRobot.txt",
    "WarehouseThreeRobotFulfillment.txt",
]

METHODS = {
    "bfs": {
        "label": "Breadth First Search",
        "env": {},
        "description": "Part 1 baseline. Complete and shortest in action count for unit-cost STRIPS actions.",
    },
    "astar": {
        "label": "A Star Search (relaxed-plan delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "astar"},
        "description": "Lecture-level heuristic planner using an FF-style relaxed-plan delete-relaxation heuristic.",
    },
    "astar_goal": {
        "label": "A Star Search (goal-count heuristic)",
        "env": {"SYMBOLIC_PLANNER_MODE": "astar_goal"},
        "description": "Older simple baseline using the number of unsatisfied goal facts.",
    },
    "astar_hadd": {
        "label": "A Star Search (hadd delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "astar_hadd"},
        "description": "Delete-relaxation heuristic using additive fact costs.",
    },
    "optimal": {
        "label": "A Star Search (hmax delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "optimal"},
        "description": "Recommended optimal heuristic mode: admissible hmax delete-relaxation heuristic.",
    },
    "strong": {
        "label": "Strong planner: A Star Search (hadd delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "strong"},
        "description": "Recommended homework mode: complete heuristic search with the strongest tested lecture-level heuristic in this repo.",
    },
    "weighted_ff": {
        "label": "Weighted A Star Search (relaxed-plan delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "weighted_ff", "SYMBOLIC_PLANNER_WEIGHT": "5"},
        "description": "Faster satisficing mode inspired by weighted heuristic search planners.",
    },
    "greedy_ff": {
        "label": "Greedy Best First Search (relaxed-plan delete relaxation)",
        "env": {"SYMBOLIC_PLANNER_MODE": "greedy_ff"},
        "description": "Fast satisficing mode inspired by FF-style greedy heuristic search.",
    },
}

STACK_SPACING = 1.65
_SEARCH_PROFILE_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


@dataclass(frozen=True)
class Condition:
    name: str
    args: Tuple[str, ...]
    truth: bool = True

    def fact(self) -> str:
        return f"{self.name}({','.join(self.args)})"

    def text(self) -> str:
        return self.fact() if self.truth else f"!{self.fact()}"


@dataclass
class ActionSchema:
    name: str
    params: Tuple[str, ...]
    preconditions: List[Condition]
    effects: List[Condition]


@dataclass
class Environment:
    name: str
    symbols: List[str]
    initial: Set[str]
    goals: Set[str]
    actions: List[ActionSchema]


@dataclass
class GroundedAction:
    name: str
    args: Tuple[str, ...]

    def text(self) -> str:
        return f"{self.name}({','.join(self.args)})"


ACTION_RE = re.compile(r"([A-Z][A-Za-z0-9_]*)\(([A-Za-z0-9_,]*)\)")
COND_RE = re.compile(r"(!?[A-Z][A-Za-z0-9_]*)\(([A-Za-z0-9_,]*)\)")


def compact(line: str) -> str:
    return re.sub(r"\s+", "", line)


def split_symbols(text: str) -> List[str]:
    return [item for item in text.split(",") if item]


def parse_conditions(line: str) -> List[Condition]:
    conditions: List[Condition] = []
    for match in COND_RE.finditer(line):
        name = match.group(1)
        truth = True
        if name.startswith("!"):
            name = name[1:]
            truth = False
        conditions.append(Condition(name, tuple(split_symbols(match.group(2))), truth))
    return conditions


def parse_environment(path: Path) -> Environment:
    raw_lines = [compact(line) for line in path.read_text().splitlines()]
    lines = [line for line in raw_lines if line]

    symbols: List[str] = []
    initial: Set[str] = set()
    goals: Set[str] = set()
    actions: List[ActionSchema] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        lower = line.lower()
        if lower.startswith("symbols:"):
            symbols = split_symbols(line.split(":", 1)[1])
        elif lower.startswith("initialconditions:"):
            initial = {cond.fact() for cond in parse_conditions(line) if cond.truth}
        elif lower.startswith("goalconditions:"):
            goals = {cond.fact() for cond in parse_conditions(line) if cond.truth}
        elif lower == "actions:":
            index += 1
            while index < len(lines):
                action_match = ACTION_RE.fullmatch(lines[index])
                if not action_match:
                    raise ValueError(f"Expected action definition in {path.name}: {lines[index]}")
                name = action_match.group(1)
                params = tuple(split_symbols(action_match.group(2)))

                index += 1
                if index >= len(lines) or not lines[index].lower().startswith("preconditions:"):
                    raise ValueError(f"Missing preconditions for action {name} in {path.name}")
                preconditions = parse_conditions(lines[index])

                index += 1
                if index >= len(lines) or not lines[index].lower().startswith("effects:"):
                    raise ValueError(f"Missing effects for action {name} in {path.name}")
                effects = parse_conditions(lines[index])

                actions.append(ActionSchema(name, params, preconditions, effects))
                index += 1
            break
        index += 1

    return Environment(path.name, symbols, initial, goals, actions)


def parse_grounded_action(text: str) -> GroundedAction:
    match = ACTION_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Could not parse grounded action: {text}")
    return GroundedAction(match.group(1), tuple(split_symbols(match.group(2))))


def extract_plan(stdout: str) -> List[GroundedAction]:
    if "Plan:" not in stdout:
        return []
    plan_text = stdout.split("Plan:", 1)[1]
    actions: List[GroundedAction] = []
    for line in plan_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for match in ACTION_RE.finditer(line):
            actions.append(parse_grounded_action(match.group(0)))
    return actions


def extract_stats(stderr: str) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    for line in stderr.splitlines():
        if not line.startswith("Planner:"):
            continue
        pieces = [piece.strip() for piece in line.split("|")]
        stats["planner_name"] = pieces[0].split(":", 1)[1].strip()
        for piece in pieces[1:]:
            if "=" not in piece:
                continue
            key, value = [item.strip() for item in piece.split("=", 1)]
            if value in {"yes", "no"}:
                stats[key] = value == "yes"
            else:
                try:
                    stats[key] = int(value)
                except ValueError:
                    try:
                        stats[key] = float(value)
                    except ValueError:
                        stats[key] = value
    return stats


def bind_condition(condition: Condition, binding: Dict[str, str]) -> Condition:
    return Condition(
        condition.name,
        tuple(binding.get(arg, arg) for arg in condition.args),
        condition.truth,
    )


def condition_holds(state: Set[str], condition: Condition) -> bool:
    present = condition.fact() in state
    return present if condition.truth else not present


def find_schema(env: Environment, action: GroundedAction) -> ActionSchema | None:
    for schema in env.actions:
        if schema.name == action.name and len(schema.params) == len(action.args):
            return schema
    return None


def replay_plan(env: Environment, plan: Sequence[GroundedAction]) -> Dict[str, object]:
    state = set(env.initial)
    trace: List[Dict[str, object]] = [
        {
            "step": 0,
            "action": "START",
            "valid": True,
            "preconditions": [],
            "effects": [],
            "missing_preconditions": [],
            "added": [],
            "deleted": [],
            "goals_satisfied": sorted(env.goals & state),
            "goal_count": len(env.goals & state),
            "state": sorted(state),
        }
    ]

    valid = True
    errors: List[str] = []

    for step_number, action in enumerate(plan, start=1):
        schema = find_schema(env, action)
        if schema is None:
            valid = False
            errors.append(f"No schema found for {action.text()}")
            trace.append(
                {
                    "step": step_number,
                    "action": action.text(),
                    "valid": False,
                    "preconditions": [],
                    "effects": [],
                    "missing_preconditions": ["schema_not_found"],
                    "added": [],
                    "deleted": [],
                    "goals_satisfied": sorted(env.goals & state),
                    "goal_count": len(env.goals & state),
                    "state": sorted(state),
                }
            )
            continue

        binding = dict(zip(schema.params, action.args))
        grounded_preconditions = [bind_condition(cond, binding) for cond in schema.preconditions]
        missing = [cond.text() for cond in grounded_preconditions if not condition_holds(state, cond)]

        grounded_effects = [bind_condition(effect, binding) for effect in schema.effects]
        before = set(state)

        if missing:
            valid = False
            errors.append(f"{action.text()} missing preconditions: {', '.join(missing)}")
        else:
            for effect in grounded_effects:
                if not effect.truth:
                    state.discard(effect.fact())
            for effect in grounded_effects:
                if effect.truth:
                    state.add(effect.fact())

        after = set(state)
        trace.append(
                {
                    "step": step_number,
                    "action": action.text(),
                    "valid": not missing,
                    "preconditions": [cond.text() for cond in grounded_preconditions],
                    "effects": [effect.text() for effect in grounded_effects],
                    "missing_preconditions": missing,
                    "added": sorted(after - before),
                    "deleted": sorted(before - after),
                "goals_satisfied": sorted(env.goals & state),
                "goal_count": len(env.goals & state),
                "state": sorted(state),
            }
        )

    missing_goals = sorted(env.goals - state)
    if missing_goals:
        valid = False
        errors.append("Final state misses goals: " + ", ".join(missing_goals))

    return {
        "valid": valid,
        "errors": errors,
        "missing_goals": missing_goals,
        "final_state": sorted(state),
        "trace": trace,
    }


def _fallback_search_profile(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    reason: str,
) -> Dict[str, object]:
    plan_length = len(plan)
    total_expanded = int(stats.get("expanded", 0) or 0)
    total_generated = int(stats.get("generated", 0) or 0)
    denominator = max(1, plan_length)
    steps = list(range(plan_length + 1))
    expanded_by_step = [round(total_expanded * step / denominator) for step in steps]
    generated_by_step = [round(total_generated * step / denominator) for step in steps]
    return {
        "environment": env.name,
        "method": method,
        "source": "linear fallback",
        "reason": reason,
        "solved": bool(stats.get("solved", False)),
        "expanded_total": total_expanded,
        "generated_total": total_generated,
        "elapsed_ms": float(stats.get("time_ms", 0.0) or 0.0),
        "records": [],
        "steps": steps,
        "expanded_by_step": expanded_by_step,
        "generated_by_step": generated_by_step,
        "actions": ["START"] + [action.text() for action in plan],
    }


def search_effort_profile(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
) -> Dict[str, object]:
    """Reconstruct how much search work was needed before each plan prefix.

    The C++ executable prints aggregate counts only. For visualization we replay
    the same STRIPS search in Python and map the returned solution path back to
    the expansion timeline. The aggregate C++ numbers are still shown in the run
    metrics; this profile supplies the per-step curve used in GIF side panels.
    """
    cache_key = (env.name, method + "|" + "|".join(action.text() for action in plan))
    if cache_key in _SEARCH_PROFILE_CACHE:
        return _SEARCH_PROFILE_CACHE[cache_key]

    try:
        import importlib

        vse = importlib.import_module("visualize_search_exploration")
        expansion_hint = int(stats.get("expanded", 0) or 0)
        generated_hint = int(stats.get("generated", 0) or 0)
        if expansion_hint > 2500 or generated_hint > 1500:
            raise RuntimeError("search replay skipped for large run; using aggregate planner counters")
        result = vse.simulate_search(env, method, max(100000, expansion_hint + 1000))
        solution_order = vse.ordered_solution_path(result.nodes, result.goal_index)
        if not solution_order:
            raise RuntimeError("search simulation did not find a solution path")

        expansion_rank = {
            node_index: rank + 1 for rank, node_index in enumerate(result.expanded_order)
        }

        def expanded_when_generated(sequence: int) -> int:
            if sequence <= 0:
                return 0
            for row in result.records:
                if int(row.get("generated", 0)) >= sequence:
                    return int(row.get("expanded", 0))
            return len(result.expanded_order)

        def generated_at_expanded(expanded_count: int) -> int:
            if expanded_count <= 0:
                return 0
            latest = 0
            for row in result.records:
                latest = int(row.get("generated", latest))
                if int(row.get("expanded", 0)) >= expanded_count:
                    return latest
            return max(0, len(result.nodes) - 1)

        plan_node_count = len(plan) + 1
        path_nodes = list(solution_order[:plan_node_count])
        if len(path_nodes) < plan_node_count:
            path_nodes.extend([solution_order[-1]] * (plan_node_count - len(path_nodes)))

        expanded_by_step: List[int] = []
        generated_by_step: List[int] = []
        for step_index, node_index in enumerate(path_nodes):
            node = result.nodes[node_index]
            if step_index == 0:
                expanded_by_step.append(0)
                generated_by_step.append(0)
                continue

            reached_after = expansion_rank.get(node_index)
            if reached_after is None:
                reached_after = expanded_when_generated(node.sequence)
            if step_index == len(path_nodes) - 1:
                reached_after = max(reached_after, len(result.expanded_order))

            expanded_by_step.append(int(reached_after))
            generated_by_step.append(generated_at_expanded(int(reached_after)))

        profile = {
            "environment": env.name,
            "method": method,
            "source": "python search replay",
            "solved": result.solved,
            "expanded_total": len(result.expanded_order),
            "generated_total": max(0, len(result.nodes) - 1),
            "elapsed_ms": result.elapsed_ms,
            "records": result.records,
            "steps": list(range(len(path_nodes))),
            "expanded_by_step": expanded_by_step,
            "generated_by_step": generated_by_step,
            "actions": ["START"] + [action.text() for action in plan],
        }
    except Exception as exc:
        profile = _fallback_search_profile(env, method, plan, stats, str(exc))

    _SEARCH_PROFILE_CACHE[cache_key] = profile
    return profile


def draw_search_effort_axis(ax, profile: Dict[str, object], current_step: int) -> None:
    steps = list(profile.get("steps", []))
    expanded = [int(value) for value in profile.get("expanded_by_step", [])]
    generated = [int(value) for value in profile.get("generated_by_step", [])]
    if not steps or not expanded:
        ax.axis("off")
        return

    current_step = max(0, min(current_step, len(steps) - 1))
    ax.set_title("Search Effort Before Each Plan Step", fontsize=11, fontweight="bold", color="#0f172a")
    ax.plot(steps, expanded, color="#2563eb", linewidth=2.4, marker="o", label="expanded states")
    ax.axvline(current_step, color="#16a34a", linewidth=1.8, alpha=0.9)
    ax.scatter([current_step], [expanded[current_step]], color="#16a34a", s=70, zorder=5)
    ax.set_xlabel("plan step")
    ax.set_ylabel("states")
    ax.set_xlim(-0.2, max(steps) + 0.2)
    ymax = max(max(expanded), 1)
    ax.set_ylim(0, ymax * 1.18 + 1)
    ax.grid(True, alpha=0.24)
    ax.legend(fontsize=7, loc="upper left")
    action = str(profile.get("actions", [""])[current_step])
    if len(action) > 34:
        action = action[:31] + "..."
    detail = [
        f"current step: {current_step}/{max(steps)}",
        f"expanded by then: {expanded[current_step]}",
        f"generated by then: {generated[current_step] if generated else 0}",
        f"total expanded: {profile.get('expanded_total')}",
        f"action: {action}",
    ]
    ax.text(
        0.03,
        0.05,
        "\n".join(detail),
        transform=ax.transAxes,
        fontsize=7.6,
        color="#111827",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.30", alpha=0.94),
    )


def pil_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_search_effort_pil(
    draw: ImageDraw.ImageDraw,
    profile: Dict[str, object],
    current_step: int,
    box: Tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    title_font = pil_font(15, True)
    body_font = pil_font(11)
    steps = list(profile.get("steps", []))
    expanded = [int(value) for value in profile.get("expanded_by_step", [])]
    if not steps or not expanded:
        return

    current_step = max(0, min(current_step, len(steps) - 1))
    draw.rounded_rectangle(box, radius=7, fill="#ffffff", outline="#cbd5e1")
    draw.text((x0 + 12, y0 + 10), "Search Effort", font=title_font, fill="#0f172a")
    plot = (x0 + 42, y0 + 52, x1 - 18, y1 - 42)
    px0, py0, px1, py1 = plot
    draw.line((px0, py1, px1, py1), fill="#334155", width=2)
    draw.line((px0, py0, px0, py1), fill="#334155", width=2)
    ymax = max(max(expanded), 1)
    xmax = max(max(steps), 1)

    points: List[Tuple[int, int]] = []
    for step, value in zip(steps, expanded):
        px = px0 + int((px1 - px0) * step / xmax)
        py = py1 - int((py1 - py0) * value / ymax)
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill="#2563eb", width=3)
    for px, py in points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#2563eb")

    cx, cy = points[current_step]
    draw.line((cx, py0, cx, py1), fill="#16a34a", width=2)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill="#16a34a", outline="#14532d")
    draw.text((px0 - 36, py0 - 5), str(ymax), font=body_font, fill="#475569")
    draw.text((px0 - 8, py1 + 8), "0", font=body_font, fill="#475569")
    draw.text((px1 - 12, py1 + 8), str(xmax), font=body_font, fill="#475569")
    draw.text(
        (x0 + 12, y1 - 31),
        f"step {current_step}/{xmax}: {expanded[current_step]} expanded, total {profile.get('expanded_total')}",
        font=body_font,
        fill="#111827",
    )


def ensure_build() -> None:
    subprocess.run(
        ["cmake", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--config", "Release"],
        check=True,
        cwd=PROJECT_ROOT,
    )


def run_planner(env_name: str, method: str) -> Tuple[str, str, List[GroundedAction], Dict[str, object]]:
    run_env = os.environ.copy()
    run_env.pop("SYMBOLIC_PLANNER_MODE", None)
    run_env.pop("SYMBOLIC_PLANNER_HEURISTIC", None)
    run_env.pop("SYMBOLIC_PLANNER_WEIGHT", None)
    run_env.update(METHODS[method]["env"])

    completed = subprocess.run(
        [str(PLANNER_EXE), env_name],
        cwd=PROJECT_ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    stats = extract_stats(completed.stderr)
    stats["returncode"] = completed.returncode
    return completed.stdout, completed.stderr, extract_plan(completed.stdout), stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_outputs(
    env: Environment,
    method: str,
    stdout: str,
    stderr: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> None:
    run_dir = RUNS_ROOT / env.name.replace(".txt", "") / method
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    (run_dir / "plan.txt").write_text("\n".join(action.text() for action in plan) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", stats)
    write_json(run_dir / "trace.json", replay)


def wrap_join(items: Iterable[str], width: int = 54, limit: int = 18) -> str:
    lines: List[str] = []
    for item in items:
        wrapped = textwrap.wrap(item, width=width) or [item]
        lines.extend(wrapped)
        if len(lines) >= limit:
            lines = lines[:limit]
            lines.append("...")
            break
    return "\n".join(lines)


def draw_text_box(ax, title: str, body: str, color: str = "#111827") -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold", va="top", color="#0f172a")
    ax.text(0.0, 0.92, body, fontsize=8.5, va="top", color=color, family="DejaVu Sans Mono", linespacing=1.25)


def fact_condition(fact: str) -> Condition | None:
    match = COND_RE.fullmatch(fact)
    if not match:
        return None
    name = match.group(1)
    truth = True
    if name.startswith("!"):
        name = name[1:]
        truth = False
    return Condition(name, tuple(split_symbols(match.group(2))), truth)


def is_block_stack_domain(env: Environment) -> bool:
    facts = set(env.initial) | set(env.goals)
    return any(fact.startswith("On(") for fact in facts)


def object_kinds_from_state(state: Iterable[str]) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for fact in state:
        condition = fact_condition(fact)
        if condition is None or len(condition.args) != 1:
            continue
        if condition.name == "Triangle":
            kinds[condition.args[0]] = "triangle"
        elif condition.name == "Block":
            kinds.setdefault(condition.args[0], "block")
    return kinds


def object_inventory_text(state: Iterable[str]) -> str:
    objects = sorted(object_kinds_from_state(state), key=natural_key)
    return f"{len(objects)} objects: " + ", ".join(objects)


def on_relations_from_state(state: Iterable[str]) -> Dict[str, str]:
    relations: Dict[str, str] = {}
    for fact in state:
        condition = fact_condition(fact)
        if condition is None or condition.name != "On" or len(condition.args) != 2:
            continue
        obj, support = condition.args
        relations[obj] = support
    return relations


def goal_constraint_state(env: Environment) -> Set[str]:
    """Build a renderable partial state from goal On facts plus object type facts."""
    state = set(env.goals)
    type_facts = [fact for fact in env.initial if fact.startswith("Block(") or fact.startswith("Triangle(")]
    state.update(type_facts)
    return state


def on_fact_lines(state: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for fact in state:
        condition = fact_condition(fact)
        if condition and condition.name == "On" and len(condition.args) == 2:
            lines.append(f"{condition.args[0]} on {condition.args[1]}")
    return sorted(lines, key=natural_key)


def visual_slot_order(env: Environment) -> List[str]:
    """Keep initial table stacks first, then reserve fixed slots for future roots."""
    kinds = object_kinds_from_state(env.initial)
    on_map = on_relations_from_state(env.initial)
    roots = sorted([obj for obj, support in on_map.items() if support == "Table"], key=natural_key)
    remaining = sorted([obj for obj in kinds if obj not in roots], key=natural_key)
    return roots + remaining


def visual_slot_order_for_trace(env: Environment, trace: Sequence[Dict[str, object]]) -> List[str]:
    order: List[str] = []
    for step in trace:
        on_map = on_relations_from_state(step["state"])
        for root in sorted([obj for obj, support in on_map.items() if support == "Table"], key=natural_key):
            if root not in order:
                order.append(root)
    if not order:
        return visual_slot_order(env)
    return order


def on_summary_text(state: Iterable[str], width: int = 48, limit: int = 4) -> str:
    facts = on_fact_lines(state)
    text = "; ".join(facts) if facts else "none"
    lines = textwrap.wrap(text, width=width) or [text]
    if len(lines) > limit:
        lines = lines[:limit] + ["..."]
    return "\n".join(lines)


def block_color(name: str, kind: str) -> str:
    palette = {
        "A": "#1d4ed8",
        "B": "#ea580c",
        "C": "#16a34a",
        "D": "#7c3aed",
        "E": "#dc2626",
        "B0": "#1d4ed8",
        "B1": "#ea580c",
        "B2": "#16a34a",
        "B3": "#7c3aed",
        "B4": "#0891b2",
        "T0": "#e11d48",
        "T1": "#facc15",
    }
    if kind == "triangle":
        return palette.get(name, "#eab308")
    return palette.get(name, "#38bdf8")


def save_stable_gif(image_paths: Sequence[Path], out_path: Path, duration: int) -> None:
    """Save a GIF with one shared palette so object colors do not shift per frame."""
    if not image_paths:
        return

    with Image.open(image_paths[0]) as first_image:
        width, height = first_image.size
    palette_width = max(1, width // 4)
    palette_height = max(1, height // 4)
    sheet = Image.new("RGB", (palette_width, palette_height * len(image_paths)), "#eef2f7")
    for idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            thumb = source.convert("RGB").resize((palette_width, palette_height))
        sheet.paste(thumb, (0, idx * palette_height))

    quantize_method = getattr(Image, "Quantize", Image).MEDIANCUT
    dither_none = getattr(getattr(Image, "Dither", Image), "NONE", 0)
    palette = sheet.quantize(colors=256, method=quantize_method, dither=dither_none)
    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            rgb_image = source.convert("RGB")
        frames.append(rgb_image.quantize(palette=palette, dither=dither_none))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
        optimize=False,
    )
    for frame in frames:
        frame.close()
    palette.close()
    sheet.close()
    gc.collect()


def natural_key(text: str) -> Tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def stack_columns_from_state(state: Iterable[str], include_loose: bool = True) -> Tuple[List[List[str]], Dict[str, str]]:
    on_map = on_relations_from_state(state)
    kinds = object_kinds_from_state(state)
    children: Dict[str, List[str]] = {}
    objects = set(kinds) | set(on_map)
    for obj, support in on_map.items():
        objects.add(support)
        children.setdefault(support, []).append(obj)
    for support in children:
        children[support].sort(key=natural_key)

    roots = sorted([obj for obj, support in on_map.items() if support == "Table"], key=natural_key)

    # Partial goal descriptions often omit the table support. Treat objects whose
    # support is not itself placed on anything as the root support for a target stack.
    if not roots:
        supported_objects = set(on_map)
        candidate_roots = sorted(
            [support for support in set(on_map.values()) if support != "Table" and support not in supported_objects],
            key=natural_key,
        )
        roots = candidate_roots

    columns: List[List[str]] = []
    seen: Set[str] = set()

    def add_chain(root: str) -> None:
        column: List[str] = []
        current = root
        while current and current not in seen and current != "Table":
            seen.add(current)
            column.append(current)
            next_children = [child for child in children.get(current, []) if child not in seen]
            current = next_children[0] if next_children else ""
        if column:
            columns.append(column)

    for root in roots:
        add_chain(root)

    # Keep unplaced objects visible in current states, but do not clutter partial
    # goal scenes with every object that is irrelevant to the goal constraints.
    placed = {obj for column in columns for obj in column}
    if include_loose:
        loose = sorted([obj for obj in objects if obj != "Table" and obj not in placed and obj in kinds], key=natural_key)
        for obj in loose:
            columns.append([obj])

    return columns, kinds


def slot_columns_from_state(
    state: Iterable[str],
    slot_order: Sequence[str],
    include_loose: bool = True,
    hidden_objects: Iterable[str] | None = None,
) -> Tuple[List[List[str]], Dict[str, str]]:
    columns, kinds = stack_columns_from_state(state, include_loose=include_loose)
    hidden = set(hidden_objects or [])
    slot_index = {obj: idx for idx, obj in enumerate(slot_order)}
    fixed_columns: List[List[str]] = [[] for _ in slot_order]
    overflow: List[List[str]] = []

    for column in columns:
        visible_column = [obj for obj in column if obj not in hidden]
        if not visible_column:
            continue
        root = visible_column[0]
        if root in slot_index:
            fixed_columns[slot_index[root]] = visible_column
        else:
            overflow.append(visible_column)

    for column in overflow:
        fixed_columns.append(column)

    return fixed_columns, kinds


def object_pose_map(state: Iterable[str], slot_order: Sequence[str]) -> Dict[str, Tuple[float, float, str]]:
    columns, kinds = slot_columns_from_state(state, slot_order)
    poses: Dict[str, Tuple[float, float, str]] = {}
    for col_idx, column in enumerate(columns):
        x = col_idx * STACK_SPACING
        for level, obj in enumerate(column):
            kind = kinds.get(obj, "triangle" if obj.startswith("T") else "block")
            poses[obj] = (x, level * 0.84, kind)
    return poses


def set_3d_axes(
    ax,
    title: str,
    columns: Sequence[Sequence[str]],
    slot_count: int | None = None,
    max_height_override: int | None = None,
) -> None:
    max_height = max_height_override if max_height_override is not None else max((len(column) for column in columns), default=1)
    width = max(2, slot_count if slot_count is not None else len(columns))
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    right = (width - 1) * STACK_SPACING + 0.9
    ax.set_xlim(-0.95, right)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.12, max_height * 0.88 + 1.0)
    ax.view_init(elev=22, azim=-47)
    ax.set_box_aspect((max(2.4, width * 1.25), 1.6, max(1.8, max_height + 0.7)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)


def draw_table(ax, column_count: int) -> None:
    count = max(1, column_count)
    left = -0.68
    right = (count - 1) * STACK_SPACING + 0.68
    width = right - left
    ax.bar3d(left, -0.56, -0.08, width, 1.12, 0.08, color="#a8a29e", edgecolor="#57534e", alpha=0.98, shade=False)
    ax.text((left + right) / 2, -0.76, 0.03, "Table", fontsize=8, color="#292524", ha="center")

    for col_idx in range(count):
        x = col_idx * STACK_SPACING
        ax.plot([x - 0.52, x + 0.52], [-0.47, -0.47], [0.012, 0.012], color="#57534e", linewidth=0.8, alpha=0.7)
        ax.plot([x - 0.52, x + 0.52], [0.47, 0.47], [0.012, 0.012], color="#57534e", linewidth=0.8, alpha=0.7)
        ax.plot([x - 0.52, x - 0.52], [-0.47, 0.47], [0.012, 0.012], color="#57534e", linewidth=0.8, alpha=0.7)
        ax.plot([x + 0.52, x + 0.52], [-0.47, 0.47], [0.012, 0.012], color="#57534e", linewidth=0.8, alpha=0.7)


def draw_block(ax, name: str, x: float, z: float, color: str, alpha: float = 0.96) -> None:
    ax.bar3d(x - 0.43, -0.36, z, 0.86, 0.72, 0.82, color=color, edgecolor="#020617", linewidth=0.75, alpha=alpha, shade=False)


def draw_triangle_prism(ax, name: str, x: float, z: float, color: str, alpha: float = 0.96) -> None:
    y0, y1 = -0.36, 0.36
    verts = [
        [(x - 0.48, y0, z), (x + 0.48, y0, z), (x, y0, z + 0.86)],
        [(x - 0.48, y1, z), (x + 0.48, y1, z), (x, y1, z + 0.86)],
        [(x - 0.48, y0, z), (x + 0.48, y0, z), (x + 0.48, y1, z), (x - 0.48, y1, z)],
        [(x - 0.48, y0, z), (x, y0, z + 0.86), (x, y1, z + 0.86), (x - 0.48, y1, z)],
        [(x + 0.48, y0, z), (x, y0, z + 0.86), (x, y1, z + 0.86), (x + 0.48, y1, z)],
    ]
    poly = Poly3DCollection(verts, facecolors=color, edgecolors="#020617", linewidths=0.8, alpha=alpha)
    ax.add_collection3d(poly)


def annotate_object_label(ax, name: str, x: float, z: float) -> None:
    x2, y2, _ = proj3d.proj_transform(x, -0.42, z + 0.93, ax.get_proj())
    ax.annotate(
        name,
        xy=(x2, y2),
        xycoords="data",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#020617",
        bbox=dict(facecolor="white", edgecolor="#020617", boxstyle="round,pad=0.16", alpha=0.96),
        zorder=1000,
    )


def render_stack_scene(
    ax,
    state: Iterable[str],
    title: str,
    highlight: Iterable[str] | None = None,
    ghost: bool = False,
    slot_order: Sequence[str] | None = None,
    max_height: int | None = None,
    hidden_objects: Iterable[str] | None = None,
    floating_object: Tuple[str, float, float, str] | None = None,
) -> None:
    if slot_order is None:
        columns, kinds = stack_columns_from_state(state, include_loose=not ghost)
        slot_count = len(columns)
    else:
        columns, kinds = slot_columns_from_state(
            state,
            slot_order,
            include_loose=not ghost,
            hidden_objects=hidden_objects,
        )
        slot_count = len(columns)
    highlight_set = set(highlight or [])
    set_3d_axes(ax, title, columns, slot_count=slot_count, max_height_override=max_height)
    draw_table(ax, slot_count)
    label_anchors: List[Tuple[str, float, float]] = []

    for col_idx, column in enumerate(columns):
        x = col_idx * STACK_SPACING
        for level, obj in enumerate(column):
            kind = kinds.get(obj, "triangle" if obj.startswith("T") else "block")
            color = block_color(obj, kind)
            alpha = 0.48 if ghost else 0.98
            if obj in highlight_set:
                alpha = 1.0
            z = level * 0.84
            if kind == "triangle":
                draw_triangle_prism(ax, obj, x, z, color, alpha)
            else:
                draw_block(ax, obj, x, z, color, alpha)
            label_anchors.append((obj, x, z))
            if obj in highlight_set:
                ax.scatter([x], [0.0], [z + 1.16], color="#f59e0b", marker="*", s=130, depthshade=False)

    if floating_object is not None:
        obj, x, z, kind = floating_object
        color = block_color(obj, kind)
        if kind == "triangle":
            draw_triangle_prism(ax, obj, x, z, color, 1.0)
        else:
            draw_block(ax, obj, x, z, color, 1.0)
        ax.scatter([x], [0.0], [z + 1.16], color="#f59e0b", marker="*", s=150, depthshade=False)
        label_anchors.append((obj, x, z))

    for obj, x, z in label_anchors:
        annotate_object_label(ax, obj, x, z)


def moved_object_from_action(action_text: str) -> str | None:
    match = ACTION_RE.fullmatch(action_text)
    if not match:
        return None
    args = split_symbols(match.group(2))
    return args[0] if args else None


def render_blockworld_3d_panel(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_block_stack_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "blockworld_3d_panel.png"
    trace = replay["trace"]
    final_state = set(trace[-1]["state"])
    slot_order = visual_slot_order_for_trace(env, trace)
    max_stack_height = max(
        (max((len(column) for column in slot_columns_from_state(step["state"], slot_order)[0]), default=1) for step in trace),
        default=1,
    )
    fixed_height = max_stack_height + 0.8

    fig = plt.figure(figsize=(18, 10), facecolor="#eef2f7")
    fig.suptitle(
        f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | 3D Symbolic Stack Plan",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.58], hspace=0.18, wspace=0.05)

    render_stack_scene(
        fig.add_subplot(grid[0, 0], projection="3d"),
        env.initial,
        "Start State From Initial Facts",
        slot_order=slot_order,
        max_height=fixed_height,
    )
    render_stack_scene(
        fig.add_subplot(grid[0, 1], projection="3d"),
        final_state,
        "Final State From Planner Replay",
        slot_order=slot_order,
        max_height=fixed_height,
    )

    ax_goal_facts = fig.add_subplot(grid[0, 2])
    ax_goal_facts.axis("off")
    goal_lines = [f"[{'x' if goal in final_state else ' '}] {goal}" for goal in sorted(env.goals)]
    extra_on = [line for line in on_fact_lines(final_state) if f"On({line.replace(' on ', ',')})" not in env.goals]
    goal_body = "\n".join(
        [
            "Goal facts are constraints, not a complete target scene.",
            "The final state is correct when every listed fact is checked.",
            "",
            "Required goal facts:",
            *goal_lines,
            "",
            "Other final On facts:",
            wrap_join(extra_on or ["none"], width=50, limit=8),
        ]
    )
    ax_goal_facts.text(0.0, 1.0, "Exact Goal Fact Check", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_goal_facts.text(
        0.0,
        0.90,
        goal_body,
        fontsize=9.6,
        family="DejaVu Sans Mono",
        color="#111827",
        va="top",
        linespacing=1.22,
    )

    ax_timeline = fig.add_subplot(grid[1, 0:2])
    ax_timeline.axis("off")
    plan_lines = []
    for idx, action in enumerate(plan, start=1):
        step = trace[idx]
        moved = moved_object_from_action(action.text()) or ""
        mark = "OK" if step["valid"] else "BAD"
        plan_lines.append(f"{idx:02d}. {mark:3s} {action.text():24s} moved={moved}")
    ax_timeline.text(0.0, 1.0, "Grounded Action Sequence", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_timeline.text(
        0.0,
        0.90,
        "\n".join(plan_lines),
        fontsize=9,
        family="DejaVu Sans Mono",
        color="#111827",
        va="top",
    )

    ax_metrics = fig.add_subplot(grid[1, 2])
    ax_metrics.axis("off")
    final_facts = set(trace[-1]["state"])
    goal_lines = [f"[{'x' if goal in final_facts else ' '}] {goal}" for goal in sorted(env.goals)]
    metrics = "\n".join(
        [
            f"Replay valid: {replay['valid']}",
            f"Plan length: {len(plan)}",
            f"Expanded states: {stats.get('expanded')}",
            f"Generated states: {stats.get('generated')}",
            f"Grounded actions: {stats.get('grounded_actions')}",
            f"Runtime ms: {float(stats.get('time_ms', 0.0)):.3f}",
            f"Scene objects: {object_inventory_text(final_facts)}",
            "",
            "Goal facts:",
            *goal_lines,
        ]
    )
    ax_metrics.text(0.0, 1.0, "Planner Check", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_metrics.text(0.0, 0.90, metrics, fontsize=9.4, family="DejaVu Sans Mono", color="#111827", va="top")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_blockworld_3d_animation(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_block_stack_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    frames_dir = vis_dir / "_blockworld_3d_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()
    out_path = vis_dir / "blockworld_3d_animation.gif"

    trace = replay["trace"]
    search_profile = search_effort_profile(env, method, plan, stats)
    image_paths: List[Path] = []
    slot_order = visual_slot_order_for_trace(env, trace)
    max_stack_height = max(
        (max((len(column) for column in slot_columns_from_state(step["state"], slot_order)[0]), default=1) for step in trace),
        default=1,
    )
    fixed_height = max_stack_height + 0.8

    def write_frame(
        frame_index: int,
        title_suffix: str,
        step: Dict[str, object],
        state: Iterable[str],
        moved: str | None,
        hidden: Iterable[str] | None = None,
        floating: Tuple[str, float, float, str] | None = None,
        transit_label: str | None = None,
    ) -> Path:
        fig = plt.figure(figsize=(15.0, 7.5), facecolor="#eef2f7", constrained_layout=False)
        fig.suptitle(
            f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | {title_suffix}",
            fontsize=15,
            fontweight="bold",
            color="#0f172a",
            y=0.975,
        )
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 1.0, 0.72],
            left=0.035,
            right=0.985,
            bottom=0.075,
            top=0.87,
            wspace=0.14,
        )

        render_stack_scene(
            fig.add_subplot(grid[0, 0:2], projection="3d"),
            state,
            "Current 3D State",
            highlight=[moved] if moved else [],
            slot_order=slot_order,
            max_height=fixed_height,
            hidden_objects=hidden,
            floating_object=floating,
        )
        draw_search_effort_axis(
            fig.add_subplot(grid[0, 2]),
            search_profile,
            int(step["step"]),
        )

        frame_path = frames_dir / f"frame_{frame_index:03d}.png"
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        gc.collect()
        return frame_path

    frame_counter = 0
    first_step = trace[0]
    image_paths.append(
        write_frame(
            frame_counter,
            f"step 0/{len(trace) - 1}",
            first_step,
            first_step["state"],
            None,
        )
    )
    frame_counter += 1

    for step_index in range(1, len(trace)):
        before = trace[step_index - 1]
        after = trace[step_index]
        moved = moved_object_from_action(str(after["action"]))
        before_poses = object_pose_map(before["state"], slot_order)
        after_poses = object_pose_map(after["state"], slot_order)

        if moved and moved in before_poses and moved in after_poses:
            x0, z0, kind0 = before_poses[moved]
            x1, z1, kind1 = after_poses[moved]
            kind = kind0 or kind1
            for transit_idx, t in enumerate((0.50,), start=1):
                x = x0 + (x1 - x0) * t
                lift = 0.75 + 0.45 * math.sin(math.pi * t)
                z = z0 + (z1 - z0) * t + lift
                image_paths.append(
                    write_frame(
                        frame_counter,
                        f"step {step_index}/{len(trace) - 1} transit",
                        after,
                        before["state"],
                        moved,
                        hidden=[moved],
                        floating=(moved, x, z, kind),
                        transit_label="moving through continuous lift/carry frame",
                    )
                )
                frame_counter += 1

        image_paths.append(
            write_frame(
                frame_counter,
                f"step {step_index}/{len(trace) - 1}",
                after,
                after["state"],
                moved,
            )
        )
        frame_counter += 1

    save_stable_gif(image_paths, out_path, duration=520)
    return out_path


def is_fire_domain(env: Environment) -> bool:
    facts = set(env.initial) | set(env.goals)
    return any(fact.startswith("Quad(") for fact in facts) and any(fact.startswith("Rob(") for fact in facts)


def facts_named(state: Iterable[str], name: str) -> List[Condition]:
    matches: List[Condition] = []
    for fact in state:
        condition = fact_condition(fact)
        if condition and condition.name == name:
            matches.append(condition)
    return matches


def fire_actor_names(env: Environment) -> Tuple[str, str]:
    quad = "Q"
    robot = "R"
    for condition in facts_named(env.initial, "Quad"):
        if condition.args:
            quad = condition.args[0]
            break
    for condition in facts_named(env.initial, "Rob"):
        if condition.args:
            robot = condition.args[0]
            break
    return quad, robot


def fire_locations(env: Environment) -> List[str]:
    locations = sorted({condition.args[0] for condition in facts_named(env.initial, "Loc") if condition.args}, key=natural_key)
    preferred = [loc for loc in ["A", "B", "C", "D", "E", "W", "F"] if loc in locations]
    return preferred + [loc for loc in locations if loc not in preferred]


def fire_location_positions(env: Environment) -> Dict[str, Tuple[float, float]]:
    base_positions = {
        "A": (-3.0, 1.0),
        "B": (-1.75, 1.75),
        "C": (-0.65, 0.35),
        "D": (-2.45, -1.15),
        "E": (-0.75, -1.55),
        "W": (1.35, 1.25),
        "F": (2.95, -1.10),
    }
    positions: Dict[str, Tuple[float, float]] = {}
    extra_index = 0
    for loc in fire_locations(env):
        if loc in base_positions:
            positions[loc] = base_positions[loc]
            continue
        angle = 2.0 * math.pi * extra_index / max(1, len(fire_locations(env)))
        positions[loc] = (2.4 * math.cos(angle), 2.0 * math.sin(angle))
        extra_index += 1
    return positions


def fire_at_location(state: Iterable[str], actor: str) -> str | None:
    for condition in facts_named(state, "At"):
        if len(condition.args) == 2 and condition.args[0] == actor:
            return condition.args[1]
    return None


def fire_stage(state: Iterable[str], fire_loc: str = "F") -> int:
    facts = set(state)
    if f"ExtThree({fire_loc})" in facts:
        return 3
    if f"ExtTwo({fire_loc})" in facts:
        return 2
    if f"ExtOne({fire_loc})" in facts:
        return 1
    return 0


def fire_state_summary(env: Environment, state: Iterable[str]) -> List[str]:
    facts = set(state)
    quad, robot = fire_actor_names(env)
    q_loc = fire_at_location(facts, quad) or "?"
    r_loc = fire_at_location(facts, robot) or "?"
    charge = "high" if f"HighCharge({quad})" in facts else "low" if f"LowCharge({quad})" in facts else "unknown"
    tank = "full" if f"FullTank({quad})" in facts else "empty" if f"EmptyTank({quad})" in facts else "unknown"
    support = "on robot" if f"OnRob({quad})" in facts else "in air" if f"InAir({quad})" in facts else "unknown"
    stage = fire_stage(facts)
    return [
        f"Robot {robot}: {r_loc}",
        f"Quad {quad}: {q_loc}, {support}",
        f"Battery: {charge}",
        f"Tank: {tank}",
        f"Fire progress: {stage}/3",
    ]


def fire_pose_from_state(env: Environment, state: Iterable[str]) -> Tuple[Tuple[float, float, float] | None, Tuple[float, float, float] | None]:
    facts = set(state)
    quad, robot = fire_actor_names(env)
    positions = fire_location_positions(env)
    q_loc = fire_at_location(facts, quad)
    r_loc = fire_at_location(facts, robot)
    robot_pose = None
    quad_pose = None
    if r_loc and r_loc in positions:
        rx, ry = positions[r_loc]
        robot_pose = (rx, ry, 0.10)
    if q_loc and q_loc in positions:
        qx, qy = positions[q_loc]
        qz = 1.10 if f"InAir({quad})" in facts else 0.62
        quad_pose = (qx, qy, qz)
    return robot_pose, quad_pose


def set_fire_axes(ax, env: Environment, title: str) -> None:
    positions = fire_location_positions(env)
    xs = [xy[0] for xy in positions.values()] or [0.0]
    ys = [xy[1] for xy in positions.values()] or [0.0]
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.9)
    ax.set_zlim(0.0, 2.25)
    ax.view_init(elev=27, azim=-52)
    ax.set_box_aspect((max(4.2, max(xs) - min(xs) + 1.5), max(3.0, max(ys) - min(ys) + 1.5), 2.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)


def draw_location_pad(ax, x: float, y: float, label: str, color: str) -> None:
    angles = [2.0 * math.pi * idx / 40 for idx in range(41)]
    radius = 0.28
    ax.plot([x + radius * math.cos(a) for a in angles], [y + radius * math.sin(a) for a in angles], [0.015] * len(angles), color=color, linewidth=1.8)
    ax.scatter([x], [y], [0.02], color=color, s=28, depthshade=False)
    ax.text(x, y, 0.04, label, fontsize=9, fontweight="bold", color="#0f172a", ha="center", va="center")


def draw_fire_flame(ax, x: float, y: float, stage: int) -> None:
    intensity = max(0, 3 - stage)
    if intensity == 0:
        ax.scatter([x], [y], [0.20], color="#22c55e", marker="*", s=180, depthshade=False)
        ax.text(x, y, 0.46, "out", fontsize=8, fontweight="bold", color="#166534", ha="center")
        return
    colors = ["#f97316", "#ef4444", "#facc15"]
    for idx in range(intensity):
        offset = (idx - 1) * 0.10
        height = 0.48 + 0.16 * idx
        verts = [
            [(x + offset, y, 0.08), (x - 0.18 + offset, y - 0.08, 0.14), (x + offset, y, height)],
            [(x + offset, y, 0.08), (x + 0.18 + offset, y - 0.08, 0.14), (x + offset, y, height)],
            [(x + offset, y, 0.08), (x, y + 0.18, 0.14), (x + offset, y, height)],
        ]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[idx % len(colors)], edgecolors="#7f1d1d", linewidths=0.5, alpha=0.92))
    ax.text(x, y, 0.86, f"fire {stage}/3", fontsize=8, fontweight="bold", color="#7f1d1d", ha="center")


def draw_robot(ax, pose: Tuple[float, float, float], label: str) -> None:
    x, y, z = pose
    ax.bar3d(x - 0.24, y - 0.18, z, 0.48, 0.36, 0.22, color="#64748b", edgecolor="#0f172a", linewidth=0.7, shade=False)
    ax.bar3d(x - 0.18, y - 0.12, z + 0.22, 0.36, 0.24, 0.14, color="#94a3b8", edgecolor="#0f172a", linewidth=0.6, shade=False)
    for dy in (-0.22, 0.18):
        ax.plot([x - 0.24, x + 0.24], [y + dy, y + dy], [z + 0.02, z + 0.02], color="#020617", linewidth=4)
    ax.text(x, y, z + 0.50, label, fontsize=8, fontweight="bold", color="#0f172a", ha="center")


def draw_quad(ax, pose: Tuple[float, float, float], label: str, high_charge: bool, full_tank: bool) -> None:
    x, y, z = pose
    body_color = "#22c55e" if high_charge else "#ef4444"
    tank_color = "#38bdf8" if full_tank else "#f8fafc"
    ax.scatter([x], [y], [z], color=body_color, s=95, edgecolors="#020617", linewidths=0.8, depthshade=False)
    ax.scatter([x], [y], [z - 0.12], color=tank_color, s=55, edgecolors="#0369a1", linewidths=0.7, depthshade=False)
    arm = 0.38
    ax.plot([x - arm, x + arm], [y, y], [z, z], color="#0f172a", linewidth=1.5)
    ax.plot([x, x], [y - arm, y + arm], [z, z], color="#0f172a", linewidth=1.5)
    angles = [2.0 * math.pi * idx / 24 for idx in range(25)]
    for rx, ry in [(x - arm, y), (x + arm, y), (x, y - arm), (x, y + arm)]:
        ax.plot([rx + 0.11 * math.cos(a) for a in angles], [ry + 0.11 * math.sin(a) for a in angles], [z + 0.02] * len(angles), color="#334155", linewidth=0.8)
    ax.text(x, y, z + 0.25, label, fontsize=8, fontweight="bold", color="#0f172a", ha="center")


def draw_fire_scene(
    ax,
    env: Environment,
    state: Iterable[str],
    title: str,
    robot_override: Tuple[float, float, float] | None = None,
    quad_override: Tuple[float, float, float] | None = None,
    water_beam: bool = False,
) -> None:
    facts = set(state)
    quad, robot = fire_actor_names(env)
    positions = fire_location_positions(env)
    set_fire_axes(ax, env, title)
    for loc, (x, y) in positions.items():
        color = "#2563eb" if loc == "W" else "#dc2626" if loc == "F" else "#475569"
        draw_location_pad(ax, x, y, loc, color)
    if "W" in positions:
        wx, wy = positions["W"]
        ax.bar3d(wx - 0.22, wy - 0.22, 0.05, 0.44, 0.44, 0.34, color="#38bdf8", edgecolor="#0369a1", linewidth=0.6, shade=False, alpha=0.78)
        ax.text(wx, wy, 0.48, "water", fontsize=8, color="#075985", fontweight="bold", ha="center")
    if "F" in positions:
        fx, fy = positions["F"]
        draw_fire_flame(ax, fx, fy, fire_stage(facts, "F"))

    robot_pose, quad_pose = fire_pose_from_state(env, facts)
    robot_pose = robot_override or robot_pose
    quad_pose = quad_override or quad_pose
    if robot_pose:
        draw_robot(ax, robot_pose, robot)
    if quad_pose:
        draw_quad(ax, quad_pose, quad, f"HighCharge({quad})" in facts, f"FullTank({quad})" in facts)
    if water_beam and quad_pose and "F" in positions:
        qx, qy, qz = quad_pose
        fx, fy = positions["F"]
        ax.plot([qx, fx], [qy, fy], [qz - 0.10, 0.40], color="#0284c7", linewidth=2.4, alpha=0.90)
        for t in (0.25, 0.50, 0.75):
            ax.scatter([qx + (fx - qx) * t], [qy + (fy - qy) * t], [qz + (0.40 - qz) * t], color="#7dd3fc", s=22, depthshade=False)


def fire_action_name(action_text: str) -> str:
    match = ACTION_RE.fullmatch(action_text)
    return match.group(1) if match else action_text


def fire_action_rule(action_text: str) -> str:
    name = fire_action_name(action_text)
    rules = {
        "START": "Initial STRIPS facts only; no transition has been applied.",
        "MoveToLoc": "Only the mobile robot changes location while the quad is in air.",
        "MoveTogether": "Robot and quad change location together only when the quad is on the robot.",
        "TakeOffFromRob": "Quad can take off only when on robot at same location with high charge.",
        "LandOnRob": "Quad can land only where robot and quad are co-located.",
        "Charge": "Quad charges only while on the mobile robot; LowCharge becomes HighCharge.",
        "FillWater": "Tank fills only when robot and quad are together at water location W.",
        "PourOnce": "Pour requires fire location, in-air quad, full tank, and high charge; tank empties and battery becomes low.",
        "PourTwice": "Second pour additionally requires ExtOne; it advances fire progress to ExtTwo.",
        "PourThrice": "Third pour additionally requires ExtTwo; it achieves ExtThree.",
    }
    return rules.get(name, "Generic STRIPS action: all grounded preconditions must hold before effects apply.")


def interpolate_pose(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def render_fire_3d_panel(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_fire_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "fire_3d_panel.png"
    trace = replay["trace"]
    final_state = set(trace[-1]["state"])

    fig = plt.figure(figsize=(18, 10), facecolor="#eef2f7")
    fig.suptitle(
        f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | 3D Fire Mission Plan",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.58], hspace=0.18, wspace=0.14)
    draw_fire_scene(fig.add_subplot(grid[0, 0], projection="3d"), env, env.initial, "Start State From Initial Facts")
    draw_fire_scene(fig.add_subplot(grid[0, 1], projection="3d"), env, final_state, "Final State From Planner Replay")

    ax_goal = fig.add_subplot(grid[0, 2])
    ax_goal.axis("off")
    goal_lines = [f"[{'x' if goal in final_state else ' '}] {goal}" for goal in sorted(env.goals)]
    ax_goal.text(0.0, 1.0, "Exact Goal Fact Check", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_goal.text(
        0.0,
        0.90,
        "\n".join(["Every checked row must be true in the final replay state.", "", *goal_lines, "", *fire_state_summary(env, final_state)]),
        fontsize=9.5,
        family="DejaVu Sans Mono",
        color="#111827",
        va="top",
        linespacing=1.22,
    )

    ax_plan = fig.add_subplot(grid[1, 0:2])
    ax_plan.axis("off")
    plan_lines = []
    for idx, action in enumerate(plan, start=1):
        step = trace[idx]
        mark = "OK" if step["valid"] else "BAD"
        plan_lines.append(f"{idx:02d}. {mark:3s} {action.text()}")
    ax_plan.text(0.0, 1.0, "Grounded Action Sequence", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_plan.text(0.0, 0.90, wrap_join(plan_lines, width=82, limit=22), fontsize=9, family="DejaVu Sans Mono", color="#111827", va="top")

    ax_metrics = fig.add_subplot(grid[1, 2])
    ax_metrics.axis("off")
    metrics = "\n".join(
        [
            f"Replay valid: {replay['valid']}",
            f"Plan length: {len(plan)}",
            f"Expanded states: {stats.get('expanded')}",
            f"Generated states: {stats.get('generated')}",
            f"Grounded actions: {stats.get('grounded_actions')}",
            f"Runtime ms: {float(stats.get('time_ms', 0.0)):.3f}",
            "",
            "Final mission state:",
            *fire_state_summary(env, final_state),
        ]
    )
    ax_metrics.text(0.0, 1.0, "Planner Check", fontsize=12, fontweight="bold", color="#0f172a", va="top")
    ax_metrics.text(0.0, 0.90, metrics, fontsize=9.5, family="DejaVu Sans Mono", color="#111827", va="top")

    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    return out_path


def render_fire_3d_animation(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_fire_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    frames_dir = vis_dir / "_fire_3d_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()
    out_path = vis_dir / "fire_3d_animation.gif"
    trace = replay["trace"]
    search_profile = search_effort_profile(env, method, plan, stats)
    image_paths: List[Path] = []
    width, height = 1280, 720
    scene_bottom = 666
    positions = fire_location_positions(env)

    def font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = font(24, True)
    header_font = font(17, True)
    body_font = font(14)
    mono_font = font(13)
    small_font = font(12)

    raw_points = {loc: (x * 120.0 - y * 58.0, x * 38.0 + y * 70.0) for loc, (x, y) in positions.items()}
    min_x = min((pt[0] for pt in raw_points.values()), default=0.0)
    max_x = max((pt[0] for pt in raw_points.values()), default=1.0)
    min_y = min((pt[1] for pt in raw_points.values()), default=0.0)
    max_y = max((pt[1] for pt in raw_points.values()), default=1.0)
    offset_x = width * 0.50 - (min_x + max_x) / 2.0
    offset_y = 292.0 - (min_y + max_y) / 2.0

    def project_pose(pose: Tuple[float, float, float]) -> Tuple[int, int]:
        x, y, z = pose
        raw_x = x * 120.0 - y * 58.0
        raw_y = x * 38.0 + y * 70.0
        return int(raw_x + offset_x), int(raw_y + offset_y - z * 78.0)

    def project_loc(loc: str, z: float = 0.0) -> Tuple[int, int]:
        x, y = positions[loc]
        return project_pose((x, y, z))

    def draw_text_lines(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], lines: Sequence[str], line_height: int, fill: str = "#111827", max_lines: int = 12) -> None:
        x, y = xy
        for idx, line in enumerate(lines[:max_lines]):
            draw.text((x, y + idx * line_height), line, font=mono_font, fill=fill)
        if len(lines) > max_lines:
            draw.text((x, y + max_lines * line_height), "...", font=mono_font, fill=fill)

    def draw_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str) -> None:
        draw.rounded_rectangle(box, radius=6, fill="#f8fafc", outline="#cbd5e1", width=1)
        draw.text((box[0] + 16, box[1] + 12), title, font=header_font, fill="#0f172a")

    def draw_pad(draw: ImageDraw.ImageDraw, loc: str) -> None:
        cx, cy = project_loc(loc)
        color = "#2563eb" if loc == "W" else "#dc2626" if loc == "F" else "#475569"
        shadow = [(cx, cy - 18), (cx + 42, cy), (cx, cy + 18), (cx - 42, cy)]
        draw.polygon(shadow, fill="#d1d5db", outline="#64748b")
        inner = [(cx, cy - 13), (cx + 32, cy), (cx, cy + 13), (cx - 32, cy)]
        draw.polygon(inner, fill="#ffffff", outline=color)
        draw.text((cx - 5 * len(loc), cy - 9), loc, font=small_font, fill="#0f172a")

    def draw_water(draw: ImageDraw.ImageDraw) -> None:
        if "W" not in positions:
            return
        cx, cy = project_loc("W")
        draw.rectangle((cx - 24, cy - 58, cx + 24, cy - 24), fill="#38bdf8", outline="#0369a1")
        draw.ellipse((cx - 24, cy - 66, cx + 24, cy - 46), fill="#7dd3fc", outline="#0369a1")
        draw.text((cx - 22, cy - 88), "water", font=small_font, fill="#075985")

    def draw_flame(draw: ImageDraw.ImageDraw, state: Iterable[str]) -> None:
        if "F" not in positions:
            return
        cx, cy = project_loc("F")
        stage = fire_stage(state, "F")
        if stage >= 3:
            draw.text((cx - 15, cy - 82), "out", font=header_font, fill="#166534")
            draw.polygon([(cx, cy - 63), (cx + 13, cy - 38), (cx + 42, cy - 35), (cx + 18, cy - 16), (cx + 27, cy + 12), (cx, cy - 2), (cx - 27, cy + 12), (cx - 18, cy - 16), (cx - 42, cy - 35), (cx - 13, cy - 38)], fill="#22c55e", outline="#166534")
            return
        for idx in range(max(1, 3 - stage)):
            shift = (idx - 1) * 10
            draw.polygon([(cx + shift, cy - 92 - idx * 8), (cx + 32 + shift, cy - 18), (cx + shift, cy + 14), (cx - 32 + shift, cy - 18)], fill="#ef4444", outline="#7f1d1d")
            draw.polygon([(cx + shift, cy - 66 - idx * 5), (cx + 18 + shift, cy - 13), (cx + shift, cy + 8), (cx - 18 + shift, cy - 13)], fill="#facc15", outline="#f97316")
        draw.text((cx - 24, cy - 120), f"fire {stage}/3", font=small_font, fill="#7f1d1d")

    def draw_robot_iso(draw: ImageDraw.ImageDraw, pose: Tuple[float, float, float], label: str) -> None:
        cx, cy = project_pose(pose)
        body = [(cx - 40, cy - 14), (cx + 36, cy - 5), (cx + 30, cy + 22), (cx - 46, cy + 12)]
        top = [(cx - 27, cy - 32), (cx + 24, cy - 24), (cx + 36, cy - 5), (cx - 40, cy - 14)]
        draw.polygon(body, fill="#64748b", outline="#0f172a")
        draw.polygon(top, fill="#94a3b8", outline="#0f172a")
        draw.ellipse((cx - 42, cy + 10, cx - 16, cy + 28), fill="#020617")
        draw.ellipse((cx + 12, cy + 15, cx + 38, cy + 33), fill="#020617")
        draw.text((cx - 8, cy - 55), label, font=header_font, fill="#0f172a")

    def draw_quad_iso(draw: ImageDraw.ImageDraw, pose: Tuple[float, float, float], label: str, state: Iterable[str]) -> None:
        facts = set(state)
        quad, _ = fire_actor_names(env)
        cx, cy = project_pose(pose)
        body_color = "#22c55e" if f"HighCharge({quad})" in facts else "#ef4444"
        tank_color = "#38bdf8" if f"FullTank({quad})" in facts else "#ffffff"
        draw.line((cx - 56, cy, cx + 56, cy), fill="#0f172a", width=3)
        draw.line((cx, cy - 44, cx, cy + 44), fill="#0f172a", width=3)
        for rx, ry in [(cx - 56, cy), (cx + 56, cy), (cx, cy - 44), (cx, cy + 44)]:
            draw.ellipse((rx - 15, ry - 8, rx + 15, ry + 8), outline="#334155", width=2)
        draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=body_color, outline="#020617", width=2)
        draw.ellipse((cx - 11, cy + 14, cx + 11, cy + 31), fill=tank_color, outline="#0369a1", width=2)
        draw.text((cx - 9, cy - 48), label, font=header_font, fill="#0f172a")

    def write_frame(
        frame_index: int,
        title_suffix: str,
        step: Dict[str, object],
        state: Iterable[str],
        robot_pose: Tuple[float, float, float] | None = None,
        quad_pose: Tuple[float, float, float] | None = None,
        water_beam: bool = False,
        phase: str = "settled symbolic state",
    ) -> Path:
        image = Image.new("RGB", (width, height), "#eef2f7")
        draw = ImageDraw.Draw(image)
        draw.text((width // 2, 18), f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | {title_suffix}", font=title_font, fill="#0f172a", anchor="ma")
        draw.text((width // 2, 58), "3D Fire Mission State", font=header_font, fill="#0f172a", anchor="ma")
        draw.rounded_rectangle((64, 88, width - 64, scene_bottom), radius=8, fill="#f8fafc", outline="#dbe4ee")

        for a, b in [("A", "B"), ("B", "C"), ("C", "W"), ("W", "F"), ("A", "D"), ("D", "E"), ("E", "W")]:
            if a in positions and b in positions:
                draw.line((*project_loc(a), *project_loc(b)), fill="#cbd5e1", width=3)
        for loc in sorted(positions, key=lambda key: project_loc(key)[1]):
            draw_pad(draw, loc)
        draw_water(draw)
        draw_flame(draw, state)

        base_robot_pose, base_quad_pose = fire_pose_from_state(env, state)
        robot_pose = robot_pose or base_robot_pose
        quad_pose = quad_pose or base_quad_pose
        quad, robot = fire_actor_names(env)
        if water_beam and quad_pose and "F" in positions:
            qx, qy = project_pose(quad_pose)
            fx, fy = project_loc("F", 0.35)
            draw.line((qx, qy + 20, fx, fy - 35), fill="#0284c7", width=4)
            for t in (0.25, 0.50, 0.75):
                draw.ellipse((qx + (fx - qx) * t - 5, qy + 20 + (fy - 35 - qy - 20) * t - 5, qx + (fx - qx) * t + 5, qy + 20 + (fy - 35 - qy - 20) * t + 5), fill="#7dd3fc")
        if robot_pose:
            draw_robot_iso(draw, robot_pose, robot)
        if quad_pose:
            draw_quad_iso(draw, quad_pose, quad, state)
        draw_search_effort_pil(draw, search_profile, int(step["step"]), (888, 104, 1202, 252))
        action_label = str(step["action"])
        if len(action_label) > 40:
            action_label = action_label[:37] + "..."
        draw.rounded_rectangle((86, 104, 440, 190), radius=7, fill="#ffffff", outline="#cbd5e1")
        draw.text((104, 118), action_label, font=header_font, fill="#0f172a")
        draw.text((104, 146), f"step {step['step']}/{len(trace) - 1} | {phase}", font=small_font, fill="#334155")
        draw.text((104, 166), " | ".join(fire_state_summary(env, state)[:2]), font=small_font, fill="#334155")

        frame_path = frames_dir / f"frame_{frame_index:03d}.png"
        image.save(frame_path)
        image.close()
        return frame_path

    frame_counter = 0
    first_step = trace[0]
    image_paths.append(write_frame(frame_counter, f"step 0/{len(trace) - 1}", first_step, first_step["state"]))
    frame_counter += 1

    for step_index in range(1, len(trace)):
        before = trace[step_index - 1]
        after = trace[step_index]
        action = fire_action_name(str(after["action"]))
        before_robot, before_quad = fire_pose_from_state(env, before["state"])
        after_robot, after_quad = fire_pose_from_state(env, after["state"])

        if action in {"MoveToLoc", "MoveTogether"} and before_robot and after_robot:
            robot_pose = interpolate_pose(before_robot, after_robot, 0.50)
            quad_pose = interpolate_pose(before_quad, after_quad, 0.50) if action == "MoveTogether" and before_quad and after_quad else None
            image_paths.append(write_frame(frame_counter, f"step {step_index}/{len(trace) - 1} transit", after, before["state"], robot_pose=robot_pose, quad_pose=quad_pose, phase="moving through route interpolation"))
            frame_counter += 1
        elif action in {"TakeOffFromRob", "LandOnRob"} and before_quad and after_quad:
            image_paths.append(write_frame(frame_counter, f"step {step_index}/{len(trace) - 1} transit", after, before["state"], quad_pose=interpolate_pose(before_quad, after_quad, 0.50), phase="vertical takeoff/landing interpolation"))
            frame_counter += 1
        elif action.startswith("Pour"):
            image_paths.append(write_frame(frame_counter, f"step {step_index}/{len(trace) - 1} pour", after, before["state"], water_beam=True, phase="water release toward fire"))
            frame_counter += 1

        image_paths.append(write_frame(frame_counter, f"step {step_index}/{len(trace) - 1}", after, after["state"]))
        frame_counter += 1

    save_stable_gif(image_paths, out_path, duration=460)
    return out_path


def is_robot_task_domain(env: Environment) -> bool:
    facts = set(env.initial) | set(env.goals)
    return (
        any(fact.startswith("Robot(") for fact in facts)
        and any(fact.startswith("Item(") for fact in facts)
        and any(fact.startswith("ItemAt(") for fact in facts)
    )


def robot_task_names(env: Environment, predicate: str) -> List[str]:
    names = {
        condition.args[0]
        for condition in facts_named(env.initial, predicate)
        if condition.args
    }
    return sorted(names, key=natural_key)


def robot_task_locations(env: Environment) -> List[str]:
    locations = {
        condition.args[0]
        for condition in facts_named(env.initial, "Loc")
        if condition.args
    }
    return sorted(locations, key=natural_key)


def robot_task_positions(env: Environment) -> Dict[str, Tuple[float, float]]:
    locations = robot_task_locations(env)
    preset = {
        "Base": (-2.8, 0.0),
        "Garage": (-2.8, 0.0),
        "Depot": (-2.8, 0.0),
        "Staging": (-2.8, 0.0),
        "Dock": (-2.8, 0.0),
        "Pharmacy": (-0.8, 1.55),
        "Ward": (1.05, 1.30),
        "Lab": (2.75, 0.0),
        "OR": (0.65, -1.55),
        "Supply": (-1.0, -1.45),
        "Shelter": (1.05, 1.30),
        "Clinic": (2.75, 0.0),
        "Command": (0.65, -1.55),
        "Aisle1": (-0.8, 1.55),
        "Aisle2": (1.05, 1.30),
        "Pack": (2.75, 0.0),
        "Quality": (0.65, -1.55),
    }
    positions: Dict[str, Tuple[float, float]] = {}
    missing = [loc for loc in locations if loc not in preset]
    for loc in locations:
        if loc in preset:
            positions[loc] = preset[loc]
    for idx, loc in enumerate(missing):
        angle = 2.0 * math.pi * idx / max(1, len(missing))
        positions[loc] = (2.4 * math.cos(angle), 1.7 * math.sin(angle))
    return positions


def robot_task_entity_location(state: Iterable[str], predicate: str, entity: str) -> str | None:
    for condition in facts_named(state, predicate):
        if len(condition.args) == 2 and condition.args[0] == entity:
            return condition.args[1]
    return None


def robot_task_carrier(state: Iterable[str], item: str) -> str | None:
    for condition in facts_named(state, "Carrying"):
        if len(condition.args) == 2 and condition.args[1] == item:
            return condition.args[0]
    return None


def robot_task_carried_item(state: Iterable[str], robot: str) -> str | None:
    for condition in facts_named(state, "Carrying"):
        if len(condition.args) == 2 and condition.args[0] == robot:
            return condition.args[1]
    return None


def robot_task_pose(env: Environment, state: Iterable[str], entity: str, entity_kind: str) -> Tuple[float, float, float] | None:
    positions = robot_task_positions(env)
    if entity_kind == "robot":
        loc = robot_task_entity_location(state, "At", entity)
        if loc in positions:
            x, y = positions[loc]
            return (x, y, 0.10)
        return None

    loc = robot_task_entity_location(state, "ItemAt", entity)
    if loc in positions:
        x, y = positions[loc]
        return (x + 0.18, y - 0.20, 0.18)
    carrier = robot_task_carrier(state, entity)
    if carrier:
        robot_pose = robot_task_pose(env, state, carrier, "robot")
        if robot_pose:
            return (robot_pose[0], robot_pose[1], 0.62)
    return None


def robot_color(name: str) -> str:
    palette = {
        "R1": "#2563eb",
        "R2": "#16a34a",
        "R3": "#f97316",
        "R4": "#7c3aed",
    }
    return palette.get(name, "#64748b")


def item_color(name: str) -> str:
    palette = {
        "MedKit": "#ef4444",
        "BloodSample": "#a855f7",
        "SterileTray": "#06b6d4",
        "Water": "#38bdf8",
        "Radio": "#facc15",
        "FoodCrate": "#84cc16",
        "PartA": "#f97316",
        "PartB": "#22c55e",
        "PartC": "#8b5cf6",
    }
    return palette.get(name, "#f59e0b")


def set_robot_task_axes(ax, env: Environment, title: str) -> None:
    positions = robot_task_positions(env)
    xs = [xy[0] for xy in positions.values()] or [0.0]
    ys = [xy[1] for xy in positions.values()] or [0.0]
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_zlim(0.0, 1.9)
    ax.view_init(elev=26, azim=-48)
    ax.set_box_aspect((max(4.4, max(xs) - min(xs) + 1.6), max(3.2, max(ys) - min(ys) + 1.6), 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)


def draw_robot_task_location(ax, x: float, y: float, label: str) -> None:
    ax.bar3d(x - 0.34, y - 0.28, 0.0, 0.68, 0.56, 0.045, color="#e2e8f0", edgecolor="#475569", linewidth=0.7, shade=False)
    ax.text(x, y, 0.075, label, fontsize=8, fontweight="bold", color="#0f172a", ha="center", va="center")


def draw_robot_task_item(ax, pose: Tuple[float, float, float], item: str, highlight: bool = False) -> None:
    x, y, z = pose
    ax.bar3d(x - 0.15, y - 0.13, z, 0.30, 0.26, 0.24, color=item_color(item), edgecolor="#0f172a", linewidth=0.7, shade=False)
    ax.text(x, y, z + 0.32, item, fontsize=7.5, fontweight="bold", color="#0f172a", ha="center")
    if highlight:
        ax.scatter([x], [y], [z + 0.43], color="#facc15", marker="*", s=115, depthshade=False)


def draw_robot_task_robot(ax, pose: Tuple[float, float, float], robot: str, carrying: str | None, highlight: bool = False) -> None:
    x, y, z = pose
    ax.bar3d(x - 0.22, y - 0.18, z, 0.44, 0.36, 0.24, color=robot_color(robot), edgecolor="#0f172a", linewidth=0.75, shade=False)
    ax.scatter([x - 0.18, x + 0.18], [y - 0.23, y - 0.23], [z + 0.04, z + 0.04], color="#020617", s=20, depthshade=False)
    ax.text(x, y, z + 0.40, robot, fontsize=8.5, fontweight="bold", color="#0f172a", ha="center")
    if carrying:
        draw_robot_task_item(ax, (x, y, z + 0.34), carrying, highlight=highlight)
    if highlight:
        ax.scatter([x], [y], [z + 0.78], color="#facc15", marker="*", s=130, depthshade=False)


def draw_robot_task_scene(
    ax,
    env: Environment,
    state: Iterable[str],
    title: str,
    robot_overrides: Dict[str, Tuple[float, float, float]] | None = None,
    item_overrides: Dict[str, Tuple[float, float, float]] | None = None,
    highlight_robot: str | None = None,
    highlight_item: str | None = None,
) -> None:
    facts = set(state)
    positions = robot_task_positions(env)
    robots = robot_task_names(env, "Robot")
    items = robot_task_names(env, "Item")
    robot_overrides = robot_overrides or {}
    item_overrides = item_overrides or {}
    set_robot_task_axes(ax, env, title)

    def spread_offset(index: int, count: int, radius: float) -> Tuple[float, float]:
        if count <= 1:
            return (0.0, 0.0)
        angle = 2.0 * math.pi * index / count + math.pi / 4.0
        return (radius * math.cos(angle), radius * math.sin(angle))

    ordered_locations = sorted(positions, key=lambda loc: positions[loc][1])
    for first, second in zip(ordered_locations, ordered_locations[1:]):
        x0, y0 = positions[first]
        x1, y1 = positions[second]
        ax.plot([x0, x1], [y0, y1], [0.03, 0.03], color="#cbd5e1", linewidth=1.2)
    for loc, (x, y) in positions.items():
        draw_robot_task_location(ax, x, y, loc)

    robot_pose_lookup: Dict[str, Tuple[float, float, float]] = {}
    robots_by_loc: Dict[str, List[str]] = {}
    for robot in robots:
        loc = robot_task_entity_location(facts, "At", robot)
        if loc in positions:
            robots_by_loc.setdefault(loc, []).append(robot)
    for loc, loc_robots in robots_by_loc.items():
        loc_robots.sort(key=natural_key)
        base_x, base_y = positions[loc]
        for idx, robot in enumerate(loc_robots):
            dx, dy = spread_offset(idx, len(loc_robots), 0.20)
            robot_pose_lookup[robot] = (base_x + dx, base_y + dy, 0.10)

    item_pose_lookup: Dict[str, Tuple[float, float, float]] = {}
    items_by_loc: Dict[str, List[str]] = {}
    for item in items:
        loc = robot_task_entity_location(facts, "ItemAt", item)
        if loc in positions:
            items_by_loc.setdefault(loc, []).append(item)
    for loc, loc_items in items_by_loc.items():
        loc_items.sort(key=natural_key)
        base_x, base_y = positions[loc]
        for idx, item in enumerate(loc_items):
            dx, dy = spread_offset(idx, len(loc_items), 0.26)
            item_pose_lookup[item] = (base_x + dx, base_y + dy, 0.18)

    carried_items = {item for item in items if robot_task_carrier(facts, item)}
    for item in items:
        if item in item_overrides:
            draw_robot_task_item(ax, item_overrides[item], item, highlight=item == highlight_item)
            continue
        if item in carried_items:
            continue
        item_pose = item_pose_lookup.get(item) or robot_task_pose(env, facts, item, "item")
        if item_pose:
            draw_robot_task_item(ax, item_pose, item, highlight=item == highlight_item)

    for robot in robots:
        pose = robot_overrides.get(robot) or robot_pose_lookup.get(robot) or robot_task_pose(env, facts, robot, "robot")
        if not pose:
            continue
        carried = robot_task_carried_item(facts, robot)
        if carried in item_overrides:
            carried = None
        draw_robot_task_robot(ax, pose, robot, carried, highlight=robot == highlight_robot)


def robot_task_action_args(action_text: str) -> Tuple[str, Tuple[str, ...]]:
    match = ACTION_RE.fullmatch(action_text)
    if not match:
        return action_text, ()
    return match.group(1), tuple(split_symbols(match.group(2)))


def robot_task_summary(env: Environment, state: Iterable[str]) -> List[str]:
    facts = set(state)
    lines: List[str] = []
    for robot in robot_task_names(env, "Robot"):
        loc = robot_task_entity_location(facts, "At", robot) or "?"
        carried = robot_task_carried_item(facts, robot) or "empty"
        lines.append(f"{robot}: {loc}, carrying {carried}")
    for item in robot_task_names(env, "Item"):
        loc = robot_task_entity_location(facts, "ItemAt", item)
        carrier = robot_task_carrier(facts, item)
        if loc:
            item_state = loc
        elif carrier:
            item_state = "with " + carrier
        else:
            item_state = "?"
        lines.append(f"{item}: {item_state}")
    return lines


def robot_task_theme(env: Environment) -> str:
    lower = env.name.lower()
    if "hospital" in lower:
        return "hospital"
    if "warehouse" in lower:
        return "warehouse"
    if "disaster" in lower:
        return "disaster"
    return "generic"


def robot_task_short_item(item: str) -> str:
    names = {
        "MedKit": "Med",
        "BloodSample": "Blood",
        "SterileTray": "Tray",
        "FoodCrate": "Food",
        "PartA": "A",
        "PartB": "B",
        "PartC": "C",
    }
    return names.get(item, item)


def robot_task_map_spec(env: Environment) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]]]:
    theme = robot_task_theme(env)
    if theme == "hospital":
        return (
            {
                "Garage": (0.12, 0.75),
                "Pharmacy": (0.34, 0.44),
                "Ward": (0.54, 0.28),
                "Lab": (0.79, 0.42),
                "OR": (0.60, 0.74),
            },
            [("Garage", "Pharmacy"), ("Pharmacy", "Ward"), ("Ward", "Lab"), ("Ward", "OR"), ("Pharmacy", "OR")],
        )
    if theme == "warehouse":
        return (
            {
                "Dock": (0.12, 0.72),
                "Aisle1": (0.34, 0.34),
                "Aisle2": (0.55, 0.34),
                "Pack": (0.80, 0.62),
                "Quality": (0.66, 0.80),
            },
            [("Dock", "Aisle1"), ("Aisle1", "Aisle2"), ("Aisle2", "Pack"), ("Aisle2", "Quality"), ("Pack", "Quality")],
        )
    if theme == "disaster":
        return (
            {
                "Staging": (0.12, 0.74),
                "Depot": (0.33, 0.48),
                "Shelter": (0.56, 0.30),
                "Clinic": (0.80, 0.48),
                "Command": (0.63, 0.77),
            },
            [("Staging", "Depot"), ("Depot", "Shelter"), ("Shelter", "Clinic"), ("Depot", "Command"), ("Command", "Clinic")],
        )

    locations = robot_task_locations(env)
    center_x, center_y = 0.50, 0.52
    radius_x, radius_y = 0.34, 0.26
    coords: Dict[str, Tuple[float, float]] = {}
    for idx, loc in enumerate(locations):
        angle = 2.0 * math.pi * idx / max(1, len(locations))
        coords[loc] = (center_x + radius_x * math.cos(angle), center_y + radius_y * math.sin(angle))
    edges = list(zip(locations, locations[1:]))
    return coords, edges


def robot_task_screen_positions(env: Environment, box: Tuple[int, int, int, int]) -> Dict[str, Tuple[int, int]]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    spec, _ = robot_task_map_spec(env)
    positions: Dict[str, Tuple[int, int]] = {}
    for loc in robot_task_locations(env):
        rx, ry = spec.get(loc, (0.5, 0.5))
        positions[loc] = (int(x0 + rx * width), int(y0 + ry * height))
    return positions


def draw_centered_text(draw: ImageDraw.ImageDraw, center: Tuple[int, int], text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = center[0] - (bbox[2] - bbox[0]) // 2
    y = center[1] - (bbox[3] - bbox[1]) // 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_robot_task_route(
    draw: ImageDraw.ImageDraw,
    positions: Dict[str, Tuple[int, int]],
    a: str,
    b: str,
    color: str,
    width: int,
) -> None:
    if a not in positions or b not in positions:
        return
    x0, y0 = positions[a]
    x1, y1 = positions[b]
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    for t in (0.25, 0.50, 0.75):
        x = int(x0 + (x1 - x0) * t)
        y = int(y0 + (y1 - y0) * t)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_robot_task_background(
    draw: ImageDraw.ImageDraw,
    env: Environment,
    box: Tuple[int, int, int, int],
    positions: Dict[str, Tuple[int, int]],
    active_route: Tuple[str, str] | None,
    label_font,
    small_font,
) -> None:
    theme = robot_task_theme(env)
    _, edges = robot_task_map_spec(env)
    x0, y0, x1, y1 = box
    fill = "#f8fafc"
    if theme == "warehouse":
        fill = "#f1f5f9"
    elif theme == "disaster":
        fill = "#ecfdf5"
    draw.rounded_rectangle(box, radius=8, fill=fill, outline="#dbe4ee")

    if theme == "hospital":
        for y in range(y0 + 24, y1, 36):
            draw.line((x0 + 12, y, x1 - 12, y), fill="#e5e7eb", width=1)
        for x in range(x0 + 24, x1, 48):
            draw.line((x, y0 + 12, x, y1 - 12), fill="#eef2f7", width=1)
    elif theme == "warehouse":
        for x in range(x0 + 28, x1, 64):
            draw.line((x, y0 + 16, x, y1 - 16), fill="#dbe4ee", width=1)
        for y in range(y0 + 30, y1, 56):
            draw.line((x0 + 16, y, x1 - 16, y), fill="#e2e8f0", width=1)
    elif theme == "disaster":
        for x in range(x0 + 40, x1, 92):
            draw.ellipse((x - 5, y1 - 54, x + 5, y1 - 42), fill="#bbf7d0")

    for a, b in edges:
        robot_task_route_color = "#cbd5e1" if theme != "disaster" else "#d6a65f"
        robot_task_route_width = 16 if theme == "hospital" else 12
        draw_robot_task_route(draw, positions, a, b, robot_task_route_color, robot_task_route_width)
    if active_route:
        draw_robot_task_route(draw, positions, active_route[0], active_route[1], "#2563eb", 8)

    for loc in sorted(positions, key=lambda item: positions[item][1]):
        cx, cy = positions[loc]
        if theme == "hospital":
            colors = {
                "Garage": "#e2e8f0",
                "Pharmacy": "#dcfce7",
                "Ward": "#dbeafe",
                "Lab": "#f3e8ff",
                "OR": "#fee2e2",
            }
            draw.rounded_rectangle((cx - 62, cy - 34, cx + 62, cy + 34), radius=6, fill=colors.get(loc, "#ffffff"), outline="#64748b", width=2)
            if loc in {"Ward", "OR"}:
                draw.line((cx - 11, cy - 15, cx - 11, cy - 2), fill="#dc2626", width=5)
                draw.line((cx - 17, cy - 8, cx - 5, cy - 8), fill="#dc2626", width=5)
            elif loc == "Pharmacy":
                draw.rectangle((cx - 20, cy - 17, cx - 4, cy - 2), fill="#16a34a", outline="#14532d")
            elif loc == "Lab":
                draw.ellipse((cx - 19, cy - 17, cx - 5, cy - 3), fill="#a855f7", outline="#6b21a8")
            draw_centered_text(draw, (cx + 6, cy + 14), loc, label_font, "#0f172a")
        elif theme == "warehouse":
            if loc.startswith("Aisle"):
                for offset in (-28, 0, 28):
                    draw.rounded_rectangle((cx - 64, cy + offset - 9, cx + 64, cy + offset + 7), radius=3, fill="#94a3b8", outline="#475569")
                    for bin_x in range(cx - 50, cx + 52, 25):
                        draw.rectangle((bin_x, cy + offset - 6, bin_x + 15, cy + offset + 5), fill="#f59e0b", outline="#92400e")
                draw_centered_text(draw, (cx, cy - 55), loc, label_font, "#0f172a")
            else:
                pad_fill = "#dbeafe" if loc == "Dock" else "#dcfce7" if loc == "Pack" else "#fef3c7"
                draw.rounded_rectangle((cx - 64, cy - 34, cx + 64, cy + 34), radius=5, fill=pad_fill, outline="#475569", width=2)
                if loc == "Dock":
                    draw.rectangle((cx - 48, cy - 20, cx + 48, cy - 6), fill="#64748b")
                elif loc == "Pack":
                    draw.rectangle((cx - 45, cy - 18, cx + 45, cy + 2), fill="#92400e")
                    draw.line((cx - 40, cy + 12, cx + 40, cy + 12), fill="#334155", width=4)
                elif loc == "Quality":
                    draw.ellipse((cx - 26, cy - 24, cx + 26, cy + 24), outline="#ca8a04", width=4)
                draw_centered_text(draw, (cx, cy + 18), loc, label_font, "#0f172a")
        elif theme == "disaster":
            if loc in {"Shelter", "Clinic", "Command"}:
                tent_color = "#fde68a" if loc == "Shelter" else "#fecaca" if loc == "Clinic" else "#bfdbfe"
                draw.polygon([(cx, cy - 42), (cx + 60, cy + 20), (cx - 60, cy + 20)], fill=tent_color, outline="#475569")
                draw.line((cx, cy - 42, cx, cy + 20), fill="#475569", width=2)
            else:
                draw.rounded_rectangle((cx - 60, cy - 30, cx + 60, cy + 30), radius=5, fill="#fef3c7", outline="#92400e", width=2)
            draw_centered_text(draw, (cx, cy + 39), loc, label_font, "#0f172a")
        else:
            draw.rounded_rectangle((cx - 58, cy - 30, cx + 58, cy + 30), radius=6, fill="#ffffff", outline="#64748b", width=2)
            draw_centered_text(draw, (cx, cy), loc, label_font, "#0f172a")


def spread_screen_offsets(index: int, count: int, radius: int) -> Tuple[int, int]:
    if count <= 1:
        return (0, 0)
    angle = 2.0 * math.pi * index / count + math.pi / 5.0
    return (int(radius * math.cos(angle)), int(radius * math.sin(angle)))


def draw_robot_task_item_icon(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    item: str,
    font,
    highlight: bool = False,
) -> None:
    cx, cy = center
    color = item_color(item)
    if highlight:
        draw.ellipse((cx - 33, cy - 33, cx + 33, cy + 33), outline="#facc15", width=5)
    if item == "MedKit":
        draw.rounded_rectangle((cx - 22, cy - 17, cx + 22, cy + 17), radius=4, fill="#ef4444", outline="#7f1d1d", width=2)
        draw.line((cx, cy - 10, cx, cy + 10), fill="#ffffff", width=5)
        draw.line((cx - 10, cy, cx + 10, cy), fill="#ffffff", width=5)
    elif item == "BloodSample":
        draw.rounded_rectangle((cx - 10, cy - 25, cx + 10, cy + 20), radius=5, fill="#f5f3ff", outline="#6b21a8", width=2)
        draw.rectangle((cx - 8, cy - 2, cx + 8, cy + 18), fill="#a855f7")
    elif item == "SterileTray":
        draw.rounded_rectangle((cx - 26, cy - 11, cx + 26, cy + 13), radius=8, fill="#cffafe", outline="#0891b2", width=2)
        draw.line((cx - 17, cy - 2, cx + 17, cy - 2), fill="#06b6d4", width=3)
    elif item == "Radio":
        draw.rounded_rectangle((cx - 18, cy - 17, cx + 18, cy + 18), radius=4, fill="#facc15", outline="#854d0e", width=2)
        draw.line((cx + 13, cy - 18, cx + 31, cy - 38), fill="#334155", width=3)
        draw.ellipse((cx - 7, cy - 6, cx + 7, cy + 8), fill="#0f172a")
    elif item == "Water":
        draw.rectangle((cx - 17, cy - 20, cx + 17, cy + 18), fill="#38bdf8", outline="#075985", width=2)
        draw.ellipse((cx - 17, cy - 26, cx + 17, cy - 10), fill="#7dd3fc", outline="#075985", width=2)
    else:
        draw.polygon([(cx - 24, cy - 16), (cx + 10, cy - 26), (cx + 28, cy - 8), (cx - 6, cy + 4)], fill=color, outline="#0f172a")
        draw.polygon([(cx - 24, cy - 16), (cx - 6, cy + 4), (cx - 6, cy + 27), (cx - 24, cy + 8)], fill=color, outline="#0f172a")
        draw.polygon([(cx - 6, cy + 4), (cx + 28, cy - 8), (cx + 28, cy + 15), (cx - 6, cy + 27)], fill="#fde68a", outline="#0f172a")
        draw_centered_text(draw, (cx + 2, cy + 5), robot_task_short_item(item), font, "#0f172a")
    draw_centered_text(draw, (cx, cy + 41), robot_task_short_item(item), font, "#0f172a")


def draw_robot_task_robot_icon(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    robot: str,
    carrying: str | None,
    label_font,
    item_font,
    highlight: bool = False,
    highlight_item: str | None = None,
) -> None:
    cx, cy = center
    color = robot_color(robot)
    if highlight:
        draw.ellipse((cx - 47, cy - 47, cx + 47, cy + 47), outline="#facc15", width=6)
    draw.ellipse((cx - 44, cy + 14, cx + 44, cy + 34), fill="#cbd5e1")
    body = [(cx - 44, cy - 16), (cx + 35, cy - 8), (cx + 44, cy + 18), (cx - 34, cy + 24)]
    top = [(cx - 28, cy - 34), (cx + 27, cy - 28), (cx + 35, cy - 8), (cx - 44, cy - 16)]
    draw.polygon(body, fill=color, outline="#0f172a")
    draw.polygon(top, fill="#bfdbfe", outline="#0f172a")
    draw.ellipse((cx - 39, cy + 18, cx - 17, cy + 36), fill="#020617")
    draw.ellipse((cx + 16, cy + 20, cx + 38, cy + 38), fill="#020617")
    draw_centered_text(draw, (cx, cy - 54), robot, label_font, "#0f172a")
    if carrying:
        draw_robot_task_item_icon(draw, (cx, cy - 18), carrying, item_font, highlight=carrying == highlight_item)


def robot_task_highlights(env: Environment, action_text: str) -> Tuple[str | None, str | None, Tuple[str, str] | None]:
    name, args = robot_task_action_args(action_text)
    robots = set(robot_task_names(env, "Robot"))
    items = set(robot_task_names(env, "Item"))
    highlight_robot = args[0] if args and args[0] in robots else None
    highlight_item = args[1] if len(args) > 1 and args[1] in items else None
    active_route = (args[1], args[2]) if name == "Drive" and len(args) >= 3 else None
    return highlight_robot, highlight_item, active_route


def robot_task_screen_pose(
    env: Environment,
    state: Iterable[str],
    entity: str,
    entity_kind: str,
    box: Tuple[int, int, int, int],
) -> Tuple[int, int] | None:
    positions = robot_task_screen_positions(env, box)
    facts = set(state)
    if entity_kind == "robot":
        loc = robot_task_entity_location(facts, "At", entity)
        return positions.get(loc) if loc else None
    loc = robot_task_entity_location(facts, "ItemAt", entity)
    if loc and loc in positions:
        return positions[loc]
    carrier = robot_task_carrier(facts, entity)
    if carrier:
        return robot_task_screen_pose(env, facts, carrier, "robot", box)
    return None


def draw_robot_task_world_pil(
    draw: ImageDraw.ImageDraw,
    env: Environment,
    state: Iterable[str],
    box: Tuple[int, int, int, int],
    label_font,
    small_font,
    robot_overrides: Dict[str, Tuple[int, int]] | None = None,
    highlight_robot: str | None = None,
    highlight_item: str | None = None,
    active_route: Tuple[str, str] | None = None,
) -> None:
    facts = set(state)
    robots = robot_task_names(env, "Robot")
    items = robot_task_names(env, "Item")
    positions = robot_task_screen_positions(env, box)
    robot_overrides = robot_overrides or {}
    draw_robot_task_background(draw, env, box, positions, active_route, label_font, small_font)

    items_by_loc: Dict[str, List[str]] = {}
    for item in items:
        if robot_task_carrier(facts, item):
            continue
        loc = robot_task_entity_location(facts, "ItemAt", item)
        if loc in positions:
            items_by_loc.setdefault(loc, []).append(item)

    item_draws: List[Tuple[int, str, Tuple[int, int]]] = []
    for loc, loc_items in items_by_loc.items():
        loc_items.sort(key=natural_key)
        cx, cy = positions[loc]
        for idx, item in enumerate(loc_items):
            dx, dy = spread_screen_offsets(idx, len(loc_items), 44)
            item_draws.append((cy + dy, item, (cx + dx + 20, cy + dy - 76)))

    robots_by_loc: Dict[str, List[str]] = {}
    for robot in robots:
        loc = robot_task_entity_location(facts, "At", robot)
        if loc in positions:
            robots_by_loc.setdefault(loc, []).append(robot)

    robot_points: Dict[str, Tuple[int, int]] = {}
    for loc, loc_robots in robots_by_loc.items():
        loc_robots.sort(key=natural_key)
        cx, cy = positions[loc]
        for idx, robot in enumerate(loc_robots):
            dx, dy = spread_screen_offsets(idx, len(loc_robots), 68)
            robot_points[robot] = (cx + dx - 8, cy + dy + 24)
    robot_points.update(robot_overrides)

    for _, item, point in sorted(item_draws):
        draw_robot_task_item_icon(draw, point, item, small_font, highlight=item == highlight_item)

    robot_draws = sorted(((point[1], robot, point) for robot, point in robot_points.items()), key=lambda row: row[0])
    for _, robot, point in robot_draws:
        carried = robot_task_carried_item(facts, robot)
        draw_robot_task_robot_icon(
            draw,
            point,
            robot,
            carried,
            label_font,
            small_font,
            highlight=robot == highlight_robot,
            highlight_item=highlight_item,
        )


def render_robot_task_3d_panel(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_robot_task_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "robot_task_3d_panel.png"
    trace = replay["trace"]
    final_state = set(trace[-1]["state"])
    search_profile = search_effort_profile(env, method, plan, stats)

    width, height = 1600, 820
    image = Image.new("RGB", (width, height), "#eef2f7")
    draw = ImageDraw.Draw(image)
    title_font = pil_font(26, True)
    header_font = pil_font(18, True)
    label_font = pil_font(14, True)
    small_font = pil_font(12)

    title = f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | Multi-Robot Task Plan"
    draw.text((width // 2, 24), title, font=title_font, fill="#0f172a", anchor="ma")
    start_box = (54, 96, 770, 650)
    final_box = (830, 96, 1546, 650)
    draw.text((start_box[0], 66), "Start State", font=header_font, fill="#0f172a")
    draw.text((final_box[0], 66), "Final Replay State", font=header_font, fill="#0f172a")
    draw_robot_task_world_pil(draw, env, env.initial, start_box, label_font, small_font)
    draw_robot_task_world_pil(draw, env, final_state, final_box, label_font, small_font)
    draw_search_effort_pil(draw, search_profile, len(plan), (1192, 118, 1528, 276))

    goals_met = len(set(env.goals) & final_state)
    footer = (
        f"Replay valid: {replay['valid']}    Plan length: {len(plan)}    "
        f"Goals reached: {goals_met}/{len(env.goals)}    "
        f"Expanded: {stats.get('expanded')}    Generated: {stats.get('generated')}    "
        f"Runtime ms: {float(stats.get('time_ms', 0.0)):.3f}"
    )
    draw.rounded_rectangle((54, 684, 1546, 760), radius=7, fill="#ffffff", outline="#cbd5e1")
    draw.text((80, 710), footer, font=header_font, fill="#0f172a")

    image.save(out_path)
    image.close()
    return out_path


def render_robot_task_3d_animation(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path | None:
    if not is_robot_task_domain(env):
        return None

    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    frames_dir = vis_dir / "_robot_task_3d_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()
    out_path = vis_dir / "robot_task_3d_animation.gif"
    trace = replay["trace"]
    search_profile = search_effort_profile(env, method, plan, stats)
    image_paths: List[Path] = []

    width, height = 1280, 720
    scene_box = (54, 86, 1226, 668)
    title_font = pil_font(24, True)
    header_font = pil_font(17, True)
    label_font = pil_font(14, True)
    small_font = pil_font(12)

    def write_frame(
        frame_index: int,
        title_suffix: str,
        step: Dict[str, object],
        state: Iterable[str],
        robot_overrides: Dict[str, Tuple[int, int]] | None = None,
        highlight_robot: str | None = None,
        highlight_item: str | None = None,
        active_route: Tuple[str, str] | None = None,
        phase: str = "settled",
    ) -> Path:
        image = Image.new("RGB", (width, height), "#eef2f7")
        draw = ImageDraw.Draw(image)
        draw.text(
            (width // 2, 18),
            f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | {title_suffix}",
            font=title_font,
            fill="#0f172a",
            anchor="ma",
        )
        draw_robot_task_world_pil(
            draw,
            env,
            state,
            scene_box,
            label_font,
            small_font,
            robot_overrides=robot_overrides,
            highlight_robot=highlight_robot,
            highlight_item=highlight_item,
            active_route=active_route,
        )
        draw_search_effort_pil(draw, search_profile, int(step["step"]), (884, 104, 1208, 252))

        action_label = str(step["action"])
        if len(action_label) > 42:
            action_label = action_label[:39] + "..."
        summary_lines = robot_task_summary(env, state)
        draw.rounded_rectangle((76, 104, 474, 198), radius=7, fill="#ffffff", outline="#cbd5e1")
        draw.text((96, 118), action_label, font=header_font, fill="#0f172a")
        draw.text((96, 146), f"step {step['step']}/{len(trace) - 1} | {phase}", font=small_font, fill="#334155")
        draw.text((96, 166), " | ".join(summary_lines[:2]), font=small_font, fill="#334155")

        frame_path = frames_dir / f"frame_{frame_index:03d}.png"
        image.save(frame_path)
        image.close()
        return frame_path

    frame_counter = 0
    first_step = trace[0]
    image_paths.append(write_frame(frame_counter, f"step 0/{len(trace) - 1}", first_step, first_step["state"]))
    frame_counter += 1

    for step_index in range(1, len(trace)):
        before = trace[step_index - 1]
        after = trace[step_index]
        name, args = robot_task_action_args(str(after["action"]))
        highlight_robot, highlight_item, active_route = robot_task_highlights(env, str(after["action"]))

        if name == "Drive" and len(args) >= 3:
            robot = args[0]
            before_point = robot_task_screen_pose(env, before["state"], robot, "robot", scene_box)
            after_point = robot_task_screen_pose(env, after["state"], robot, "robot", scene_box)
            if before_point and after_point:
                mid_point = (
                    int(before_point[0] + (after_point[0] - before_point[0]) * 0.50),
                    int(before_point[1] + (after_point[1] - before_point[1]) * 0.50 + 24),
                )
                image_paths.append(
                    write_frame(
                        frame_counter,
                        f"step {step_index}/{len(trace) - 1} drive",
                        after,
                        before["state"],
                        robot_overrides={robot: mid_point},
                        highlight_robot=robot,
                        highlight_item=robot_task_carried_item(before["state"], robot),
                        active_route=active_route,
                        phase="driving",
                    )
                )
                frame_counter += 1

        image_paths.append(
            write_frame(
                frame_counter,
                f"step {step_index}/{len(trace) - 1}",
                after,
                after["state"],
                highlight_robot=highlight_robot,
                highlight_item=highlight_item,
                active_route=active_route,
            )
        )
        frame_counter += 1

    save_stable_gif(image_paths, out_path, duration=520)
    return out_path


def render_plan_panel(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path:
    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "plan_panel.png"

    trace = replay["trace"]
    progress = [step["goal_count"] for step in trace]
    steps = list(range(len(progress)))

    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    fig.suptitle(
        f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | STRIPS State-Space Trace",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.55, 1.05], height_ratios=[1.0, 0.9], hspace=0.26, wspace=0.24)

    summary = "\n".join(
        [
            f"Symbols: {', '.join(env.symbols)}",
            "",
            "Initial facts:",
            wrap_join(sorted(env.initial), width=42, limit=10),
            "",
            "Goal facts:",
            wrap_join(sorted(env.goals), width=42, limit=8),
        ]
    )
    draw_text_box(fig.add_subplot(grid[0, 0]), "Environment", summary)

    action_lines = []
    for idx, action in enumerate(plan, start=1):
        step = trace[idx]
        mark = "OK" if step["valid"] else "BAD"
        action_lines.append(f"{idx:02d}. [{mark}] {action.text()}")
    draw_text_box(
        fig.add_subplot(grid[0, 1]),
        "Grounded Plan",
        "\n".join(action_lines) if action_lines else "Already at goal.",
        color="#064e3b" if replay["valid"] else "#991b1b",
    )

    ax_progress = fig.add_subplot(grid[0, 2])
    ax_progress.set_title("Satisfied Goal Facts by Step", fontsize=12, fontweight="bold")
    ax_progress.plot(steps, progress, marker="o", color="#2563eb", linewidth=2.2)
    ax_progress.set_xlabel("Plan step")
    ax_progress.set_ylabel("Goals satisfied")
    ax_progress.set_ylim(-0.1, max(1, len(env.goals)) + 0.25)
    ax_progress.set_yticks(range(0, max(1, len(env.goals)) + 1))
    ax_progress.grid(True, alpha=0.25)
    ax_progress.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"Solved: {stats.get('solved')}",
                f"Valid replay: {replay['valid']}",
                f"Plan length: {len(plan)}",
                f"Expanded: {stats.get('expanded')}",
                f"Generated: {stats.get('generated')}",
                f"Grounded actions: {stats.get('grounded_actions')}",
                f"Time ms: {float(stats.get('time_ms', 0.0)):.3f}",
            ]
        ),
        transform=ax_progress.transAxes,
        va="top",
        fontsize=8.8,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
    )

    final_summary = "\n".join(
        [
            "Final check:",
            "all goals reached" if replay["valid"] else "NOT VALID",
            "",
            "Missing goals:",
            wrap_join(replay["missing_goals"] or ["none"], width=70, limit=5),
            "",
            "Final state facts:",
            wrap_join(replay["final_state"], width=70, limit=16),
        ]
    )
    draw_text_box(fig.add_subplot(grid[1, :]), "Replay Validation", final_summary, color="#111827")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_animation(
    env: Environment,
    method: str,
    plan: Sequence[GroundedAction],
    stats: Dict[str, object],
    replay: Dict[str, object],
) -> Path:
    vis_dir = VIS_ROOT / env.name.replace(".txt", "") / method
    frames_dir = vis_dir / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()
    out_path = vis_dir / "plan_animation.gif"

    image_paths: List[Path] = []
    trace = replay["trace"]
    total_goals = max(1, len(env.goals))

    for step in trace:
        fig = plt.figure(figsize=(12, 7), facecolor="#f8fafc")
        fig.suptitle(
            f"{env.name.replace('.txt', '')} | {METHODS[method]['label']} | step {step['step']}/{len(trace) - 1}",
            fontsize=15,
            fontweight="bold",
            color="#0f172a",
        )
        grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], hspace=0.24, wspace=0.22)

        action_body = "\n".join(
            [
                f"Action: {step['action']}",
                f"Applicable: {step['valid']}",
                "",
                "Added:",
                wrap_join(step["added"] or ["none"], width=46, limit=7),
                "",
                "Deleted:",
                wrap_join(step["deleted"] or ["none"], width=46, limit=7),
            ]
        )
        draw_text_box(fig.add_subplot(grid[0, 0]), "Transition", action_body, color="#064e3b" if step["valid"] else "#991b1b")

        goal_lines = []
        satisfied = set(step["goals_satisfied"])
        for goal in sorted(env.goals):
            goal_lines.append(f"[{'x' if goal in satisfied else ' '}] {goal}")
        draw_text_box(fig.add_subplot(grid[0, 1]), "Goals", "\n".join(goal_lines))

        ax_bar = fig.add_subplot(grid[1, 0])
        ax_bar.set_title("Progress", fontsize=12, fontweight="bold")
        ax_bar.barh(["goals"], [step["goal_count"]], color="#16a34a")
        ax_bar.set_xlim(0, total_goals)
        ax_bar.set_xlabel("satisfied goal facts")
        ax_bar.grid(True, axis="x", alpha=0.25)
        ax_bar.text(
            0.02,
            0.12,
            f"Expanded states: {stats.get('expanded')}\nGenerated states: {stats.get('generated')}\nPlan length: {len(plan)}",
            transform=ax_bar.transAxes,
            fontsize=10,
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.35"),
        )

        draw_text_box(
            fig.add_subplot(grid[1, 1]),
            "Current True Facts",
            wrap_join(step["state"], width=54, limit=19),
        )

        frame_path = frames_dir / f"frame_{int(step['step']):03d}.png"
        fig.savefig(frame_path, dpi=115, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        image_paths.append(frame_path)

    save_stable_gif(image_paths, out_path, duration=850)
    return out_path


def render_summary_panel(rows: Sequence[Dict[str, object]]) -> Path:
    out_path = VIS_ROOT / "summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#f8fafc")
    ax.axis("off")
    ax.set_title("Symbolic Planner Run Summary", fontsize=16, fontweight="bold", color="#0f172a", pad=18)

    columns = ["environment", "method", "solved", "valid_replay", "plan_length", "expanded", "generated", "time_ms"]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                row["environment"],
                row["method"],
                row["solved"],
                row["valid_replay"],
                row["plan_length"],
                row["expanded"],
                row["generated"],
                f"{float(row['time_ms']):.3f}",
            ]
        )

    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(fontweight="bold", color="#0f172a")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
        else:
            cell.set_facecolor("#ffffff")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(rows: Sequence[Dict[str, object]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "summary.csv"
    json_path = OUTPUT_ROOT / "summary.json"

    fieldnames = [
        "environment",
        "method",
        "planner_name",
        "solved",
        "valid_replay",
        "plan_length",
        "expanded",
        "generated",
        "grounded_actions",
        "time_ms",
        "raw_stdout",
        "raw_stderr",
        "plan_txt",
        "metrics_json",
        "trace_json",
        "search_profile_json",
        "plan_panel",
        "plan_animation",
        "blockworld_3d_panel",
        "blockworld_3d_animation",
        "fire_3d_panel",
        "fire_3d_animation",
        "robot_task_3d_panel",
        "robot_task_3d_animation",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    write_json(json_path, rows)


def run_all(environments: Sequence[str], methods: Sequence[str], skip_build: bool) -> List[Dict[str, object]]:
    if not skip_build:
        ensure_build()

    rows: List[Dict[str, object]] = []

    for env_name in environments:
        env = parse_environment(ENVS_DIR / env_name)
        for method in methods:
            stdout, stderr, plan, stats = run_planner(env_name, method)
            replay = replay_plan(env, plan)
            stats["plan_length_from_output"] = len(plan)
            stats["valid_replay"] = replay["valid"]
            stats["missing_goals"] = replay["missing_goals"]

            write_run_outputs(env, method, stdout, stderr, plan, stats, replay)
            run_dir = RUNS_ROOT / env.name.replace(".txt", "") / method
            search_profile_path = run_dir / "search_profile.json"
            write_json(search_profile_path, search_effort_profile(env, method, plan, stats))
            panel_path = render_plan_panel(env, method, plan, stats, replay)
            if is_fire_domain(env):
                animation_path = None
                blockworld_panel_path = None
                blockworld_animation_path = None
                robot_task_panel_path = None
                robot_task_animation_path = None
                fire_animation_path = render_fire_3d_animation(env, method, plan, stats, replay)
                fire_panel_path = render_fire_3d_panel(env, method, plan, stats, replay)
            elif is_robot_task_domain(env):
                animation_path = None
                blockworld_panel_path = None
                blockworld_animation_path = None
                fire_panel_path = None
                fire_animation_path = None
                robot_task_panel_path = render_robot_task_3d_panel(env, method, plan, stats, replay)
                robot_task_animation_path = render_robot_task_3d_animation(env, method, plan, stats, replay)
            else:
                animation_path = render_animation(env, method, plan, stats, replay)
                blockworld_panel_path = render_blockworld_3d_panel(env, method, plan, stats, replay)
                blockworld_animation_path = render_blockworld_3d_animation(env, method, plan, stats, replay)
                fire_panel_path = None
                fire_animation_path = None
                robot_task_panel_path = None
                robot_task_animation_path = None

            row = {
                "environment": env.name,
                "method": method,
                "planner_name": stats.get("planner_name", METHODS[method]["label"]),
                "solved": stats.get("solved", False),
                "valid_replay": replay["valid"],
                "plan_length": len(plan),
                "expanded": stats.get("expanded", ""),
                "generated": stats.get("generated", ""),
                "grounded_actions": stats.get("grounded_actions", ""),
                "time_ms": stats.get("time_ms", 0.0),
                "raw_stdout": str(run_dir / "stdout.txt"),
                "raw_stderr": str(run_dir / "stderr.txt"),
                "plan_txt": str(run_dir / "plan.txt"),
                "metrics_json": str(run_dir / "metrics.json"),
                "trace_json": str(run_dir / "trace.json"),
                "search_profile_json": str(search_profile_path),
                "plan_panel": str(panel_path),
                "plan_animation": str(animation_path) if animation_path else "",
                "blockworld_3d_panel": str(blockworld_panel_path) if blockworld_panel_path else "",
                "blockworld_3d_animation": str(blockworld_animation_path) if blockworld_animation_path else "",
                "fire_3d_panel": str(fire_panel_path) if fire_panel_path else "",
                "fire_3d_animation": str(fire_animation_path) if fire_animation_path else "",
                "robot_task_3d_panel": str(robot_task_panel_path) if robot_task_panel_path else "",
                "robot_task_3d_animation": str(robot_task_animation_path) if robot_task_animation_path else "",
            }
            rows.append(row)
            print(
                f"{env.name:24s} {method:6s} solved={row['solved']} "
                f"valid={row['valid_replay']} length={row['plan_length']} "
                f"expanded={row['expanded']} time_ms={float(row['time_ms']):.3f}"
            )
            plt.close("all")
            gc.collect()

    write_summary(rows)
    render_summary_panel(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and visualize symbolic planner environments.")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing build/planner executable.")
    parser.add_argument("--env", action="append", choices=ENVIRONMENTS, help="Environment file to run. Can be repeated.")
    parser.add_argument("--method", action="append", choices=sorted(METHODS), help="Planner method to run. Can be repeated.")
    args = parser.parse_args()

    environments = args.env if args.env else ENVIRONMENTS
    methods = args.method if args.method else list(METHODS.keys())
    run_all(environments, methods, args.skip_build)


if __name__ == "__main__":
    main()
