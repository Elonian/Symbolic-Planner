#!/usr/bin/env python3
"""Render search exploration, not just final-plan replay.

The plan replay GIFs show only the returned action sequence. This script shows
what the planner searched through: expanded states, generated frontier states,
and which expanded states finally ended up on the solution path.
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_and_visualize as rv


INF = 1_000_000


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
    g: int
    h: int
    sequence: int
    expanded_order: int | None = None


@dataclass
class SearchResult:
    method: str
    solved: bool
    nodes: List[SearchNode]
    expanded_order: List[int]
    frontier_order: List[int]
    goal_index: int | None
    elapsed_ms: float
    records: List[dict[str, int]]


def method_label(method: str) -> str:
    return rv.METHODS.get(method, {}).get("label", method)


def method_kind(method: str) -> str:
    if method == "bfs":
        return "bfs"
    if method in {"greedy", "gbfs", "greedy_ff"}:
        return "greedy"
    if method in {"weighted", "weighted_ff", "weighted_astar", "wastar"}:
        return "weighted"
    return "astar"


def state_key(state: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(state))


def generate_grounded_operators(env: rv.Environment) -> List[GroundedOperator]:
    operators: List[GroundedOperator] = []
    for schema in sorted(env.actions, key=lambda item: f"{item.name}({','.join(item.params)})"):
        variables: List[str] = []
        for param in schema.params:
            if param not in variables:
                variables.append(param)
        for values in product(env.symbols, repeat=len(variables)):
            binding = dict(zip(variables, values))
            action_args = tuple(binding.get(param, param) for param in schema.params)
            preconditions = [rv.bind_condition(cond, binding) for cond in schema.preconditions]
            effects = [rv.bind_condition(effect, binding) for effect in schema.effects]
            operators.append(GroundedOperator(rv.GroundedAction(schema.name, action_args), preconditions, effects))
    return sorted(operators, key=lambda op: op.action.text())


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


def goals_satisfied(state: frozenset[str], env: rv.Environment) -> bool:
    return env.goals <= set(state)


def goal_count(state: frozenset[str], env: rv.Environment) -> int:
    return len(env.goals & set(state))


def missing_goal_count(state: frozenset[str], env: rv.Environment) -> int:
    return len(env.goals - set(state))


def relaxed_plan_heuristic(state: frozenset[str], env: rv.Environment, operators: Sequence[GroundedOperator]) -> int:
    if goals_satisfied(state, env):
        return 0

    fact_layers: List[set[str]] = [set(state)]
    action_layers: List[List[int]] = []

    while not env.goals <= fact_layers[-1]:
        current = fact_layers[-1]
        next_facts = set(current)
        applicable_actions: List[int] = []
        for index, op in enumerate(operators):
            if all((not pre.truth) or pre.fact() in current for pre in op.preconditions):
                applicable_actions.append(index)
                for effect in op.effects:
                    if effect.truth:
                        next_facts.add(effect.fact())
        if next_facts == current:
            return INF
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
                return INF
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


def relaxed_cost_heuristic(
    state: frozenset[str],
    env: rv.Environment,
    operators: Sequence[GroundedOperator],
    use_max: bool,
) -> int:
    costs = {fact: 0 for fact in state}
    changed = True
    while changed:
        changed = False
        for op in operators:
            pre_cost = 0
            reachable = True
            for pre in op.preconditions:
                if not pre.truth:
                    continue
                cost = costs.get(pre.fact(), INF)
                if cost >= INF:
                    reachable = False
                    break
                pre_cost = max(pre_cost, cost) if use_max else min(INF, pre_cost + cost)
            if not reachable:
                continue
            action_cost = min(INF, pre_cost + 1)
            for effect in op.effects:
                if not effect.truth:
                    continue
                fact = effect.fact()
                if action_cost < costs.get(fact, INF):
                    costs[fact] = action_cost
                    changed = True

    total = 0
    for goal in env.goals:
        cost = costs.get(goal, INF)
        if cost >= INF:
            return INF
        total = max(total, cost) if use_max else min(INF, total + cost)
    return total


def heuristic(method: str, state: frozenset[str], env: rv.Environment, operators: Sequence[GroundedOperator]) -> int:
    if method == "astar_goal":
        return missing_goal_count(state, env)
    if method in {"optimal", "astar_hmax"}:
        return relaxed_cost_heuristic(state, env, operators, use_max=True)
    if method in {"strong", "astar_hadd", "best"}:
        return relaxed_cost_heuristic(state, env, operators, use_max=False)
    return relaxed_plan_heuristic(state, env, operators)


def reconstruct_path(nodes: Sequence[SearchNode], goal_index: int | None) -> set[int]:
    if goal_index is None:
        return set()
    path: set[int] = set()
    index = goal_index
    while index >= 0:
        path.add(index)
        index = nodes[index].parent
    return path


def make_search_record(nodes: Sequence[SearchNode], env: rv.Environment, expanded_count: int, frontier_count: int) -> dict[str, int]:
    best_goals = max((goal_count(node.state, env) for node in nodes), default=0)
    finite_h_values = [node.h for node in nodes if node.h < INF]
    min_h = min(finite_h_values, default=INF)
    return {
        "expanded": expanded_count,
        "generated": max(0, len(nodes) - 1),
        "frontier": frontier_count,
        "best_goal_count": best_goals,
        "min_h": min_h if min_h < INF else -1,
    }


def simulate_search(env: rv.Environment, method: str, max_expansions: int) -> SearchResult:
    start_time = time.perf_counter()
    operators = generate_grounded_operators(env)
    start_state = frozenset(env.initial)
    start_h = heuristic(method, start_state, env, operators)
    nodes = [SearchNode(start_state, -1, "START", 0, 0, start_h, 0)]
    expanded_order: List[int] = []
    records: List[dict[str, int]] = []
    goal_index = 0 if goals_satisfied(start_state, env) else None

    kind = method_kind(method)
    sequence = 1

    if kind == "bfs":
        open_queue: deque[int] = deque([0])
        visited = {state_key(start_state)}
        while open_queue and goal_index is None and len(expanded_order) < max_expansions:
            node_index = open_queue.popleft()
            nodes[node_index].expanded_order = len(expanded_order) + 1
            expanded_order.append(node_index)
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
                nodes.append(SearchNode(next_state, node_index, op.action.text(), current.depth + 1, current.g + 1, heuristic(method, next_state, env, operators), sequence))
                sequence += 1
                open_queue.append(child_index)
                if goals_satisfied(next_state, env):
                    goal_index = child_index
                    break
            records.append(make_search_record(nodes, env, len(expanded_order), len(open_queue)))
        frontier_order = list(open_queue)
    else:
        weight = 5 if kind == "weighted" else 1
        heap: List[Tuple[int, int, int, int]] = []
        first_priority = start_h if kind == "greedy" else weight * start_h
        heapq.heappush(heap, (first_priority, 0, 0, 0))
        best_g = {state_key(start_state): 0}

        while heap and goal_index is None and len(expanded_order) < max_expansions:
            _, neg_g, _, node_index = heapq.heappop(heap)
            current = nodes[node_index]
            current_g = -neg_g
            if best_g.get(state_key(current.state), current_g) != current_g:
                continue
            if goals_satisfied(current.state, env):
                goal_index = node_index
                break
            current.expanded_order = len(expanded_order) + 1
            expanded_order.append(node_index)
            for op in operators:
                if not applicable(current.state, op):
                    continue
                next_state = apply_operator(current.state, op)
                key = state_key(next_state)
                next_g = current.g + 1
                if best_g.get(key, INF) <= next_g:
                    continue
                h_value = heuristic(method, next_state, env, operators)
                child_index = len(nodes)
                nodes.append(SearchNode(next_state, node_index, op.action.text(), current.depth + 1, next_g, h_value, sequence))
                best_g[key] = next_g
                priority = h_value if kind == "greedy" else next_g + weight * h_value
                heapq.heappush(heap, (priority, -next_g, sequence, child_index))
                sequence += 1
            records.append(make_search_record(nodes, env, len(expanded_order), len(heap)))
        frontier_order = [item[3] for item in sorted(heap)[:36]]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(method, goal_index is not None, nodes, expanded_order, frontier_order, goal_index, elapsed_ms, records)


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


TITLE_FONT = font(24, True)
HEADER_FONT = font(17, True)
BODY_FONT = font(13)
SMALL_FONT = font(11)
MONO_FONT = font(12)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt=BODY_FONT) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fact_args(fact: str, name: str) -> Tuple[str, ...] | None:
    condition = rv.fact_condition(fact)
    if condition and condition.name == name:
        return condition.args
    return None


def draw_block_card(draw: ImageDraw.ImageDraw, state: Iterable[str], box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    state_set = set(state)
    kinds = rv.object_kinds_from_state(state_set)
    stacks, _ = rv.stack_columns_from_state(state_set)
    if not stacks:
        return
    max_height = max(len(stack) for stack in stacks)
    col_width = max(24, (x1 - x0 - 20) // max(1, len(stacks)))
    block_h = max(12, min(22, (y1 - y0 - 32) // max(1, max_height)))
    base_y = y1 - 12
    for col_idx, stack in enumerate(stacks[:6]):
        cx = x0 + 10 + col_idx * col_width + col_width // 2
        draw.line((cx - col_width // 2 + 3, base_y, cx + col_width // 2 - 3, base_y), fill="#94a3b8", width=2)
        for level, obj in enumerate(stack):
            by = base_y - (level + 1) * block_h
            kind = kinds.get(obj, "block")
            color = rv.block_color(obj, kind)
            if kind == "triangle":
                points = [(cx, by), (cx - col_width // 2 + 5, by + block_h), (cx + col_width // 2 - 5, by + block_h)]
                draw.polygon(points, fill=color, outline="#0f172a")
                draw.text((cx - 8, by + 3), obj, font=SMALL_FONT, fill="#0f172a")
            else:
                draw.rounded_rectangle((cx - col_width // 2 + 5, by, cx + col_width // 2 - 5, by + block_h - 2), radius=2, fill=color, outline="#0f172a")
                draw.text((cx - 8, by + 2), obj, font=SMALL_FONT, fill="#0f172a")


def fire_positions() -> dict[str, Tuple[int, int]]:
    return {
        "A": (22, 86),
        "B": (58, 42),
        "C": (112, 72),
        "D": (42, 132),
        "E": (100, 138),
        "W": (162, 94),
        "F": (224, 82),
    }


def fire_positions_normalized() -> dict[str, Tuple[float, float]]:
    return {
        "A": (0.08, 0.60),
        "B": (0.24, 0.28),
        "C": (0.48, 0.50),
        "D": (0.18, 0.88),
        "E": (0.43, 0.90),
        "W": (0.70, 0.62),
        "F": (0.94, 0.50),
    }


def draw_fire_card(draw: ImageDraw.ImageDraw, env: rv.Environment, state: Iterable[str], box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    facts = set(state)
    quad, robot = rv.fire_actor_names(env)
    q_loc = rv.fire_at_location(facts, quad)
    r_loc = rv.fire_at_location(facts, robot)
    support = "onR" if f"OnRob({quad})" in facts else "air" if f"InAir({quad})" in facts else "?"
    charge = "high" if f"HighCharge({quad})" in facts else "low" if f"LowCharge({quad})" in facts else "?"
    tank = "full" if f"FullTank({quad})" in facts else "empty" if f"EmptyTank({quad})" in facts else "?"
    stage = rv.fire_stage(facts)

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    summary_h = 22 if height >= 82 else 0
    diagram_h = max(1, height - summary_h)
    margin = max(4, min(18, int(min(width, diagram_h) * 0.08)))
    node_r = max(4, min(10, int(min(width, diagram_h) * 0.045)))
    label_ok = width >= 150 and diagram_h >= 88

    positions = fire_positions_normalized()

    def point(loc: str) -> Tuple[int, int]:
        px, py = positions[loc]
        return (
            int(x0 + margin + px * max(1, width - margin * 2)),
            int(y0 + margin + py * max(1, diagram_h - margin * 2)),
        )

    for a, b in [("A", "B"), ("B", "C"), ("C", "W"), ("W", "F"), ("A", "D"), ("D", "E"), ("E", "W")]:
        if a in positions and b in positions:
            ax, ay = point(a)
            bx, by = point(b)
            draw.line((ax, ay, bx, by), fill="#cbd5e1", width=max(1, node_r // 3))
    for loc in positions:
        px, py = point(loc)
        color = "#2563eb" if loc == "W" else "#dc2626" if loc == "F" else "#64748b"
        draw.ellipse((px - node_r, py - node_r, px + node_r, py + node_r), fill="#ffffff", outline=color, width=max(1, node_r // 3))
        if label_ok:
            draw.text((px - 4, py - 7), loc, font=SMALL_FONT, fill="#0f172a")
    if "F" in positions:
        fx, fy = point("F")
        fire_color = "#22c55e" if stage >= 3 else "#ef4444"
        flame_h = max(12, min(32, diagram_h // 5))
        flame_w = max(8, min(20, width // 12))
        draw.polygon((fx, fy - flame_h, fx + flame_w, fy, fx, fy + flame_h // 2, fx - flame_w, fy), fill=fire_color, outline="#7f1d1d")
        if label_ok:
            draw.text((fx - 18, fy - flame_h - 17), f"{stage}/3", font=SMALL_FONT, fill="#7f1d1d")
    if r_loc in positions:
        rx, ry = point(r_loc)
        robot_w = max(11, min(30, width // 9))
        robot_h = max(6, min(14, diagram_h // 14))
        draw.rectangle((rx - robot_w // 2, ry + node_r + 2, rx + robot_w // 2, ry + node_r + 2 + robot_h), fill="#64748b", outline="#0f172a")
        if label_ok:
            draw.text((rx - 5, ry + node_r + robot_h + 3), robot, font=SMALL_FONT, fill="#0f172a")
    if q_loc in positions:
        qx, qy = point(q_loc)
        qz = -max(14, diagram_h // 7) if support == "air" else -max(3, diagram_h // 35)
        body_color = "#22c55e" if charge == "high" else "#ef4444"
        rotor = max(9, min(18, width // 14))
        qr = max(5, min(9, node_r))
        draw.line((qx - rotor, qy + qz, qx + rotor, qy + qz), fill="#0f172a", width=max(1, node_r // 4))
        draw.line((qx, qy + qz - rotor, qx, qy + qz + rotor), fill="#0f172a", width=max(1, node_r // 4))
        draw.ellipse((qx - qr, qy + qz - qr, qx + qr, qy + qz + qr), fill=body_color, outline="#0f172a")
        if label_ok:
            draw.text((qx - 5, qy + qz - rotor - 14), quad, font=SMALL_FONT, fill="#0f172a")
    summary = f"Q:{q_loc or '?'} {support} {charge} {tank}"
    if summary_h:
        draw.text((x0 + 6, y1 - 19), summary[:34], font=SMALL_FONT, fill="#0f172a")


def draw_state_card(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    node: SearchNode,
    box: Tuple[int, int, int, int],
    label: str,
    border: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline=border, width=3)
    header = f"{label} d={node.depth} g={node.g} h={node.h} goals={goal_count(node.state, env)}/{len(env.goals)}"
    draw.text((x0 + 10, y0 + 8), header[:58], font=SMALL_FONT, fill="#0f172a")
    draw.text((x0 + 10, y0 + 25), node.action[:58], font=SMALL_FONT, fill="#334155")
    scene_box = (x0 + 8, y0 + 48, x1 - 8, y1 - 8)
    if rv.is_fire_domain(env):
        draw_fire_card(draw, env, node.state, scene_box)
    elif rv.is_block_stack_domain(env):
        draw_block_card(draw, node.state, scene_box)
    else:
        facts = sorted(node.state)
        for idx, fact in enumerate(facts[:7]):
            draw.text((scene_box[0], scene_box[1] + idx * 15), fact, font=SMALL_FONT, fill="#111827")


def truncate_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt=SMALL_FONT) -> str:
    if draw.textbbox((0, 0), text, font=fnt)[2] <= max_width:
        return text
    ellipsis = "..."
    shortened = text
    while shortened and draw.textbbox((0, 0), shortened + ellipsis, font=fnt)[2] > max_width:
        shortened = shortened[:-1]
    return shortened + ellipsis if shortened else ellipsis


def expansion_rank_by_node(result: SearchResult) -> dict[int, int]:
    return {node_index: rank + 1 for rank, node_index in enumerate(result.expanded_order)}


def sampled_expanded_nodes(result: SearchResult, max_tiles: int) -> List[int]:
    if not result.expanded_order or max_tiles <= 0:
        return []
    if len(result.expanded_order) <= max_tiles:
        return list(result.expanded_order)
    return [result.expanded_order[index] for index in frame_indices(len(result.expanded_order), max_tiles)]


def sampled_node_list(nodes: Sequence[int], max_tiles: int) -> List[int]:
    if len(nodes) <= max_tiles:
        return list(nodes)
    indices = frame_indices(len(nodes), max_tiles)
    return [nodes[index] for index in indices]


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    value: str,
    label: str,
    accent: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6, fill="#ffffff", outline="#cbd5e1")
    draw.rectangle((x0, y0, x0 + 5, y1), fill=accent)
    draw.text((x0 + 16, y0 + 13), value, font=HEADER_FONT, fill="#0f172a")
    draw.text((x0 + 16, y0 + 40), label, font=SMALL_FONT, fill="#475569")


def draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    value: int,
    maximum: int,
    fill: str,
    label: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 20), label, font=SMALL_FONT, fill="#334155")
    draw.rounded_rectangle(box, radius=5, fill="#e2e8f0", outline="#cbd5e1")
    fraction = min(1.0, max(0.0, value / max(1, maximum)))
    fill_w = int((x1 - x0) * fraction)
    if fill_w > 0:
        draw.rounded_rectangle((x0, y0, x0 + fill_w, y1), radius=5, fill=fill)
    draw.text((x1 + 10, y0 - 2), f"{value}/{maximum}", font=SMALL_FONT, fill="#0f172a")


def draw_mini_state_tile(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    node: SearchNode,
    box: Tuple[int, int, int, int],
    label: str,
    border: str,
    fill: str = "#ffffff",
    muted: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    outline = "#cbd5e1" if muted else border
    draw.rounded_rectangle(box, radius=5, fill=fill, outline=outline, width=2 if not muted else 1)
    header = truncate_to_width(draw, label, x1 - x0 - 12)
    draw.text((x0 + 6, y0 + 5), header, font=SMALL_FONT, fill="#334155" if muted else "#0f172a")
    if muted:
        draw.line((x0 + 8, y0 + 28, x1 - 8, y0 + 28), fill="#e2e8f0", width=2)
        draw.text((x0 + 8, y0 + 38), "not expanded yet", font=SMALL_FONT, fill="#94a3b8")
        return

    details = f"d{node.depth} g{node.g} h{node.h}"
    draw.text((x0 + 6, y0 + 22), details, font=SMALL_FONT, fill="#475569")
    action = truncate_to_width(draw, node.action, x1 - x0 - 12)
    draw.text((x0 + 6, y0 + 39), action, font=SMALL_FONT, fill="#475569")
    scene_box = (x0 + 6, y0 + 60, x1 - 6, y1 - 6)
    if scene_box[3] - scene_box[1] < 24:
        return
    if rv.is_fire_domain(env):
        draw_fire_card(draw, env, node.state, scene_box)
    elif rv.is_block_stack_domain(env):
        draw_block_card(draw, node.state, scene_box)
    else:
        for idx, fact in enumerate(sorted(node.state)[:4]):
            draw.text((scene_box[0], scene_box[1] + idx * 13), truncate_to_width(draw, fact, scene_box[2] - scene_box[0]), font=SMALL_FONT, fill="#111827")


def draw_expansion_grid(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    result: SearchResult,
    solution_path: set[int],
    box: Tuple[int, int, int, int],
    current_rank: int | None,
    title: str,
    max_tiles: int = 72,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 24), title, font=HEADER_FONT, fill="#0f172a")
    draw.rounded_rectangle(box, radius=8, fill="#f8fafc", outline="#cbd5e1")
    selected_nodes = sampled_expanded_nodes(result, max_tiles)
    if not selected_nodes:
        draw.text((x0 + 16, y0 + 16), "No expanded states.", font=BODY_FONT, fill="#475569")
        return

    ranks = expansion_rank_by_node(result)
    if current_rank is not None and 1 <= current_rank <= len(result.expanded_order):
        current_node = result.expanded_order[current_rank - 1]
        if current_node not in selected_nodes:
            selected_nodes.append(current_node)
            selected_nodes = sorted(set(selected_nodes), key=lambda node_index: ranks.get(node_index, INF))
            if len(selected_nodes) > max_tiles:
                selected_nodes = selected_nodes[: max_tiles - 1] + [current_node]
                selected_nodes = sorted(set(selected_nodes), key=lambda node_index: ranks.get(node_index, INF))

    available_w = max(1, x1 - x0 - 28)
    cols = max(3, min(10, available_w // 116))
    rows = math.ceil(len(selected_nodes) / cols)
    gap = 8
    tile_w = max(94, (x1 - x0 - 28 - gap * (cols - 1)) // cols)
    tile_h = max(86, min(160, (y1 - y0 - 28 - gap * (rows - 1)) // max(1, rows)))
    for item, node_index in enumerate(selected_nodes):
        row, col = divmod(item, cols)
        tx = x0 + 14 + col * (tile_w + gap)
        ty = y0 + 14 + row * (tile_h + gap)
        if ty + tile_h > y1 - 8:
            break
        rank = ranks.get(node_index, 0)
        node = result.nodes[node_index]
        future = current_rank is not None and rank > current_rank
        if future:
            draw_mini_state_tile(draw, env, node, (tx, ty, tx + tile_w, ty + tile_h), f"#{rank}", "#cbd5e1", "#ffffff", muted=True)
        else:
            if current_rank is not None and rank == current_rank:
                border = "#2563eb"
            elif node_index in solution_path:
                border = "#16a34a"
            else:
                border = "#64748b"
            label = f"expanded #{rank}"
            draw_mini_state_tile(draw, env, node, (tx, ty, tx + tile_w, ty + tile_h), label, border)


def draw_solution_strip(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    result: SearchResult,
    solution_path_ordered: List[int],
    box: Tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 24), title, font=HEADER_FONT, fill="#0f172a")
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd5e1")
    nodes = sampled_node_list(solution_path_ordered, 8)
    if not nodes:
        draw.text((x0 + 14, y0 + 14), "No solution path.", font=BODY_FONT, fill="#475569")
        return
    gap = 8
    tile_h = max(92, min(128, (y1 - y0 - 24 - gap * (len(nodes) - 1)) // max(1, len(nodes))))
    for idx, node_index in enumerate(nodes):
        ty = y0 + 12 + idx * (tile_h + gap)
        if ty + tile_h > y1 - 8:
            break
        node = result.nodes[node_index]
        label = f"path {idx + 1}/{len(solution_path_ordered)}"
        draw_mini_state_tile(draw, env, node, (x0 + 12, ty, x1 - 12, ty + tile_h), label, "#16a34a")


def draw_frontier_strip(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    result: SearchResult,
    box: Tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 24), title, font=HEADER_FONT, fill="#0f172a")
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd5e1")
    nodes = [idx for idx in result.frontier_order if idx < len(result.nodes)]
    nodes = sampled_node_list(nodes, 4)
    if not nodes:
        draw.text((x0 + 14, y0 + 14), "No frontier sample left.", font=BODY_FONT, fill="#475569")
        return
    gap = 8
    tile_h = max(92, min(126, (y1 - y0 - 24 - gap * (len(nodes) - 1)) // max(1, len(nodes))))
    for idx, node_index in enumerate(nodes):
        ty = y0 + 12 + idx * (tile_h + gap)
        if ty + tile_h > y1 - 8:
            break
        node = result.nodes[node_index]
        draw_mini_state_tile(draw, env, node, (x0 + 12, ty, x1 - 12, ty + tile_h), f"frontier #{node_index}", "#f97316")


def ordered_solution_path(nodes: Sequence[SearchNode], goal_index: int | None) -> List[int]:
    if goal_index is None:
        return []
    order: List[int] = []
    index = goal_index
    while index >= 0:
        order.append(index)
        index = nodes[index].parent
    order.reverse()
    return order


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [("#2563eb", "current expanded"), ("#16a34a", "final solution path"), ("#64748b", "expanded non-plan"), ("#f97316", "frontier/generated")]
    for idx, (color, text) in enumerate(items):
        yy = y + idx * 23
        draw.rounded_rectangle((x, yy, x + 18, yy + 14), radius=3, fill="#ffffff", outline=color, width=3)
        draw.text((x + 28, yy - 1), text, font=SMALL_FONT, fill="#0f172a")


def draw_horizontal_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [("#2563eb", "current expanded"), ("#16a34a", "final solution path"), ("#64748b", "expanded non-plan"), ("#f97316", "frontier/generated")]
    offset = 0
    for color, text in items:
        draw.rounded_rectangle((x + offset, y, x + offset + 18, y + 14), radius=3, fill="#ffffff", outline=color, width=3)
        draw.text((x + offset + 28, y - 1), text, font=SMALL_FONT, fill="#0f172a")
        offset += 230


def frame_indices(total_expanded: int, frame_count: int) -> List[int]:
    if total_expanded <= 1:
        return [0]
    if total_expanded <= frame_count:
        return list(range(total_expanded))
    return sorted(set(round(i * (total_expanded - 1) / (frame_count - 1)) for i in range(frame_count)))


def render_search_exploration(env: rv.Environment, method: str, result: SearchResult, max_frames: int = 36) -> Tuple[Path, Path]:
    vis_dir = rv.VIS_ROOT / env.name.replace(".txt", "") / method
    frames_dir = vis_dir / "_search_exploration_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    gif_path = vis_dir / "search_exploration.gif"
    panel_path = vis_dir / "search_exploration_panel.png"

    solution_path_order = ordered_solution_path(result.nodes, result.goal_index)
    solution_path = set(solution_path_order)
    expanded_set = set(result.expanded_order)
    non_plan_expanded = len(expanded_set - solution_path)
    width, height = 1700, 1080
    sample_indices = frame_indices(len(result.expanded_order), max_frames)
    frame_paths: List[Path] = []

    def draw_header(draw: ImageDraw.ImageDraw, subtitle: str) -> None:
        title = f"{env.name.replace('.txt', '')} | {method_label(method)} | search exploration"
        draw.text((width // 2, 24), title, font=TITLE_FONT, fill="#0f172a", anchor="ma")
        draw.text((width // 2, 58), subtitle, font=BODY_FONT, fill="#475569", anchor="ma")
        draw_legend(draw, 1360, 82)

    def draw_overview() -> Image.Image:
        image = Image.new("RGB", (width, height), "#eef2f7")
        draw = ImageDraw.Draw(image)
        draw_header(draw, "Static overview: what the search explored, not only the returned plan.")

        metric_y = 92
        metric_w = 205
        metric_gap = 12
        metrics = [
            (str(result.solved), "solved", "#16a34a" if result.solved else "#dc2626"),
            (str(len(result.expanded_order)), "expanded states", "#64748b"),
            (str(len(result.nodes) - 1), "generated states", "#f97316"),
            (str(non_plan_expanded), "expanded but not in plan", "#7c3aed"),
            (str(max(0, len(solution_path_order) - 1)), "returned plan length", "#16a34a"),
            (f"{result.elapsed_ms:.1f}", "python search ms", "#2563eb"),
        ]
        for idx, (value, label, accent) in enumerate(metrics):
            x0 = 40 + idx * (metric_w + metric_gap)
            draw_metric_card(draw, (x0, metric_y, x0 + metric_w, metric_y + 70), value, label, accent)

        explanation = [
            f"The large grid samples expanded states across the whole run. Gray cells are real planner work that did not appear in the final plan.",
            f"For this method: {len(result.expanded_order)} states were expanded, {len(result.nodes) - 1} states were generated, and {non_plan_expanded} expanded states were outside the returned plan.",
            "This frame is intentionally a complete summary because many PDF/report viewers show only the first frame of a GIF.",
        ]
        draw.rounded_rectangle((40, 182, 1660, 258), radius=8, fill="#ffffff", outline="#cbd5e1")
        for idx, line in enumerate(explanation):
            draw.text((60, 198 + idx * 19), line, font=BODY_FONT, fill="#111827")

        draw_expansion_grid(
            draw,
            env,
            result,
            solution_path,
            (40, 310, 1260, 1038),
            current_rank=None,
            title=f"Expanded-state sample ({min(40, len(result.expanded_order))} shown from {len(result.expanded_order)} expanded)",
            max_tiles=40,
        )
        draw_solution_strip(
            draw,
            env,
            result,
            solution_path_order,
            (1290, 310, 1660, 770),
            f"Returned plan path ({max(0, len(solution_path_order) - 1)} actions)",
        )
        draw_frontier_strip(
            draw,
            env,
            result,
            (1290, 820, 1660, 1038),
            "Generated/frontier sample",
        )
        return image

    def make_progress_image(frame_number: int, upto_expansion: int) -> Image.Image:
        image = Image.new("RGB", (width, height), "#eef2f7")
        draw = ImageDraw.Draw(image)
        current_rank = upto_expansion + 1
        current_index = result.expanded_order[upto_expansion]
        draw_header(draw, "Animated search progression: cells fill as states are expanded.")

        draw_metric_card(draw, (40, 92, 245, 162), f"{current_rank}", "expanded so far", "#2563eb")
        draw_metric_card(draw, (260, 92, 465, 162), str(len(result.expanded_order)), "total expanded", "#64748b")
        draw_metric_card(draw, (480, 92, 685, 162), str(len(result.nodes) - 1), "total generated", "#f97316")
        draw_metric_card(draw, (700, 92, 905, 162), str(non_plan_expanded), "non-plan expansions", "#7c3aed")
        draw_metric_card(draw, (920, 92, 1125, 162), str(max(0, len(solution_path_order) - 1)), "plan actions", "#16a34a")
        draw_progress_bar(
            draw,
            (40, 214, 1260, 234),
            current_rank,
            max(1, len(result.expanded_order)),
            "#2563eb",
            "Expansion progress through this search run",
        )

        current_node = result.nodes[current_index]
        draw_state_card(draw, env, current_node, (40, 270, 430, 500), f"current expanded node #{current_index}", "#2563eb")

        visible = set(result.expanded_order[:current_rank])
        recent_non_plan = [idx for idx in result.expanded_order[:current_rank] if idx not in solution_path]
        recent_solution = [idx for idx in result.expanded_order[:current_rank] if idx in solution_path]
        text_box = (460, 270, 1260, 500)
        draw.rounded_rectangle(text_box, radius=8, fill="#ffffff", outline="#cbd5e1")
        lines = [
            f"At this point {current_rank} states have been expanded.",
            f"Expanded non-plan states so far: {len(visible - solution_path)}",
            f"Expanded final-plan states so far: {len(visible & solution_path)}",
            f"Latest non-plan expanded node: #{recent_non_plan[-1] if recent_non_plan else 'none'}",
            f"Latest plan-path expanded node: #{recent_solution[-1] if recent_solution else 'none yet'}",
            "The lower grid is the important part: empty cells are future sampled expansions; filled gray/green cells are already expanded.",
        ]
        for idx, line in enumerate(lines):
            draw.text((480, 292 + idx * 28), line, font=BODY_FONT, fill="#111827")

        draw_solution_strip(
            draw,
            env,
            result,
            solution_path_order,
            (1290, 270, 1660, 725),
            "Final plan path",
        )
        draw_frontier_strip(
            draw,
            env,
            result,
            (1290, 775, 1660, 1038),
            "Frontier after search",
        )
        draw_expansion_grid(
            draw,
            env,
            result,
            solution_path,
            (40, 555, 1260, 1038),
            current_rank=current_rank,
            title="Expansion grid over time",
            max_tiles=40,
        )
        return image

    overview = draw_overview()
    overview.save(panel_path)
    first_path = frames_dir / "frame_000.png"
    overview.save(first_path)
    frame_paths.append(first_path)

    for frame_number, expansion_idx in enumerate(sample_indices):
        path = frames_dir / f"frame_{frame_number + 1:03d}.png"
        make_progress_image(frame_number + 1, expansion_idx).save(path)
        frame_paths.append(path)

    final_image = draw_overview()
    final_path = frames_dir / f"frame_{len(sample_indices) + 1:03d}.png"
    final_image.save(final_path)
    frame_paths.append(final_path)
    rv.save_stable_gif(frame_paths, gif_path, duration=760)
    return panel_path, gif_path


def draw_method_comparison_column(
    draw: ImageDraw.ImageDraw,
    env: rv.Environment,
    method: str,
    result: SearchResult,
    box: Tuple[int, int, int, int],
    common_budget: int | None,
    max_expanded: int,
) -> None:
    x0, y0, x1, y1 = box
    solution_order = ordered_solution_path(result.nodes, result.goal_index)
    solution_set = set(solution_order)
    expanded = len(result.expanded_order)
    generated = len(result.nodes) - 1
    non_plan = len(set(result.expanded_order) - solution_set)

    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((x0 + 18, y0 + 16), method_label(method), font=HEADER_FONT, fill="#0f172a")
    draw.text((x0 + 18, y0 + 43), f"expanded {expanded} | generated {generated} | non-plan {non_plan}", font=SMALL_FONT, fill="#334155")

    bar_x0, bar_x1 = x0 + 18, x1 - 18
    bar_top = y0 + 82
    draw.text((bar_x0, bar_top - 20), "Total expanded states, same scale across methods", font=SMALL_FONT, fill="#334155")
    draw.rounded_rectangle((bar_x0, bar_top, bar_x1, bar_top + 18), radius=5, fill="#e2e8f0", outline="#cbd5e1")
    total_w = int((bar_x1 - bar_x0) * expanded / max(1, max_expanded))
    draw.rounded_rectangle((bar_x0, bar_top, bar_x0 + total_w, bar_top + 18), radius=5, fill="#64748b")
    draw.text((bar_x1 - 80, bar_top + 26), f"{expanded}/{max_expanded}", font=SMALL_FONT, fill="#0f172a")

    if common_budget is None:
        current_rank = None
        status = "full search summary"
    else:
        current_rank = min(common_budget, expanded)
        if common_budget >= expanded:
            status = f"already solved by common budget {common_budget}"
        else:
            status = f"expanded {current_rank} by common budget {common_budget}"
        draw.text((bar_x0, bar_top + 26), status, font=SMALL_FONT, fill="#0f172a")
        draw.rounded_rectangle((bar_x0, bar_top + 48, bar_x1, bar_top + 62), radius=4, fill="#e2e8f0", outline="#cbd5e1")
        progress_w = int((bar_x1 - bar_x0) * current_rank / max(1, expanded))
        if progress_w > 0:
            draw.rounded_rectangle((bar_x0, bar_top + 48, bar_x0 + progress_w, bar_top + 62), radius=4, fill="#2563eb")

    draw.text((x0 + 18, y0 + 168), status, font=SMALL_FONT, fill="#334155")
    draw_metric_card(draw, (x0 + 18, y0 + 190, x0 + 154, y0 + 250), str(max(0, len(solution_order) - 1)), "plan actions", "#16a34a")
    draw_metric_card(draw, (x0 + 166, y0 + 190, x0 + 302, y0 + 250), str(non_plan), "wasted expanded", "#7c3aed")
    draw_metric_card(draw, (x0 + 314, y0 + 190, x0 + 450, y0 + 250), str(generated), "generated", "#f97316")

    draw_expansion_grid(
        draw,
        env,
        result,
        solution_set,
        (x0 + 18, y0 + 304, x1 - 18, y1 - 22),
        current_rank=current_rank,
        title="Sampled expanded states",
        max_tiles=9 if rv.is_fire_domain(env) else 18,
    )


def render_method_comparison(env: rv.Environment, results: dict[str, SearchResult], max_frames: int = 22) -> Tuple[Path, Path]:
    env_dir = rv.VIS_ROOT / env.name.replace(".txt", "")
    env_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = env_dir / "_search_method_comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    panel_path = env_dir / "search_method_comparison_panel.png"
    gif_path = env_dir / "search_method_comparison.gif"
    methods = list(results)
    width, height = 1700, 1080
    max_expanded = max((len(result.expanded_order) for result in results.values()), default=1)

    def draw_comparison(common_budget: int | None, frame_label: str) -> Image.Image:
        image = Image.new("RGB", (width, height), "#eef2f7")
        draw = ImageDraw.Draw(image)
        title = f"{env.name.replace('.txt', '')} | planner search comparison"
        draw.text((width // 2, 24), title, font=TITLE_FONT, fill="#0f172a", anchor="ma")
        draw.text((width // 2, 58), frame_label, font=BODY_FONT, fill="#475569", anchor="ma")

        explanation = (
            "All columns use the same expanded-state scale. "
            "BFS keeps expanding many gray non-plan states, while the heuristic planner reaches the plan with much less search work."
        )
        draw.rounded_rectangle((40, 94, 1660, 150), radius=8, fill="#ffffff", outline="#cbd5e1")
        draw.text((60, 113), explanation, font=BODY_FONT, fill="#111827")
        draw_horizontal_legend(draw, 40, 160)

        col_gap = 24
        col_w = (width - 80 - col_gap * (len(methods) - 1)) // max(1, len(methods))
        for idx, method in enumerate(methods):
            x0 = 40 + idx * (col_w + col_gap)
            draw_method_comparison_column(
                draw,
                env,
                method,
                results[method],
                (x0, 184, x0 + col_w, 1038),
                common_budget,
                max_expanded,
            )
        return image

    overview = draw_comparison(None, "Static side-by-side summary; this is the safest frame to put in a report.")
    overview.save(panel_path)
    frame_paths: List[Path] = []
    first_path = frames_dir / "frame_000.png"
    overview.save(first_path)
    frame_paths.append(first_path)

    targets = sorted(set(round(i * max_expanded / max(1, max_frames - 1)) for i in range(max_frames)))
    for frame_idx, target in enumerate(targets, start=1):
        frame = draw_comparison(target, f"Common search budget: {target} expanded states")
        frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path)

    final_path = frames_dir / f"frame_{len(targets) + 1:03d}.png"
    overview.save(final_path)
    frame_paths.append(final_path)
    rv.save_stable_gif(frame_paths, gif_path, duration=760)
    return panel_path, gif_path


def method_color(method: str) -> str:
    colors = {
        "bfs": "#2563eb",
        "astar_goal": "#f97316",
        "astar": "#dc2626",
        "strong": "#16a34a",
        "optimal": "#7c3aed",
        "greedy_ff": "#0891b2",
        "weighted_ff": "#a16207",
    }
    return colors.get(method, "#334155")


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    results: dict[str, SearchResult],
    box: Tuple[int, int, int, int],
    metric: str,
    title: str,
    y_label: str,
    y_max_override: int | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((x0 + 18, y0 + 14), title, font=HEADER_FONT, fill="#0f172a")
    left, top, right, bottom = x0 + 72, y0 + 58, x1 - 28, y1 - 52
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.line((left, top, left, bottom), fill="#334155", width=2)

    x_max = max((len(result.expanded_order) for result in results.values()), default=1)
    if y_max_override is None:
        y_max = max((record.get(metric, 0) for result in results.values() for record in result.records), default=1)
    else:
        y_max = y_max_override
    y_max = max(1, y_max)

    for tick in range(1, 5):
        gy = bottom - int((bottom - top) * tick / 4)
        gx = left + int((right - left) * tick / 4)
        draw.line((left, gy, right, gy), fill="#e2e8f0", width=1)
        draw.line((gx, top, gx, bottom), fill="#e2e8f0", width=1)
        draw.text((left - 48, gy - 7), str(round(y_max * tick / 4)), font=SMALL_FONT, fill="#475569")
        draw.text((gx - 12, bottom + 12), str(round(x_max * tick / 4)), font=SMALL_FONT, fill="#475569")

    draw.text((left, bottom + 32), "expanded states", font=SMALL_FONT, fill="#334155")
    draw.text((x0 + 16, top - 18), y_label, font=SMALL_FONT, fill="#334155")

    legend_x = x0 + 18
    legend_y = y1 - 32
    for method in results:
        color = method_color(method)
        draw.line((legend_x, legend_y + 7, legend_x + 22, legend_y + 7), fill=color, width=4)
        draw.text((legend_x + 28, legend_y), method_label(method), font=SMALL_FONT, fill="#0f172a")
        legend_x += 210

    for method, result in results.items():
        if not result.records:
            continue
        color = method_color(method)
        step = max(1, len(result.records) // 280)
        points: List[Tuple[int, int]] = []
        for record in result.records[::step]:
            x_value = record["expanded"]
            y_value = record.get(metric, 0)
            px = left + int((right - left) * x_value / max(1, x_max))
            py = bottom - int((bottom - top) * min(y_max, y_value) / y_max)
            points.append((px, py))
        if points:
            draw.line(points, fill=color, width=3)
            px, py = points[-1]
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="#0f172a")


def render_search_dashboard(env: rv.Environment, results: dict[str, SearchResult]) -> Path:
    env_dir = rv.VIS_ROOT / env.name.replace(".txt", "")
    env_dir.mkdir(parents=True, exist_ok=True)
    out_path = env_dir / "search_method_dashboard.png"
    width, height = 1700, 1080
    image = Image.new("RGB", (width, height), "#eef2f7")
    draw = ImageDraw.Draw(image)
    title = f"{env.name.replace('.txt', '')} | search effort dashboard"
    draw.text((width // 2, 24), title, font=TITLE_FONT, fill="#0f172a", anchor="ma")
    draw.text((width // 2, 58), "Curves show search work over time; this is the graph view that separates planners even when plan replay looks similar.", font=BODY_FONT, fill="#475569", anchor="ma")

    draw_line_chart(draw, results, (40, 100, 820, 520), "generated", "Generated states over search time", "generated")
    draw_line_chart(draw, results, (880, 100, 1660, 520), "frontier", "Frontier size over search time", "open/frontier")
    draw_line_chart(draw, results, (40, 580, 820, 1018), "best_goal_count", "Best goal satisfaction found so far", "goal facts", y_max_override=max(1, len(env.goals)))

    table_box = (880, 580, 1660, 1018)
    draw.rounded_rectangle(table_box, radius=8, fill="#ffffff", outline="#cbd5e1")
    draw.text((900, 602), "Planner summary", font=HEADER_FONT, fill="#0f172a")
    header = f"{'method':<28} {'plan':>5} {'expanded':>9} {'generated':>10} {'nonplan':>8} {'ms':>8}"
    draw.text((900, 642), header, font=MONO_FONT, fill="#334155")
    draw.line((900, 665, 1640, 665), fill="#cbd5e1", width=1)
    for idx, (method, result) in enumerate(results.items()):
        solution_order = ordered_solution_path(result.nodes, result.goal_index)
        non_plan = len(set(result.expanded_order) - set(solution_order))
        row = f"{method_label(method)[:28]:<28} {max(0, len(solution_order)-1):>5} {len(result.expanded_order):>9} {len(result.nodes)-1:>10} {non_plan:>8} {result.elapsed_ms:>8.1f}"
        draw.text((900, 682 + idx * 30), row, font=MONO_FONT, fill=method_color(method))

    draw.text((900, 830), "Interpretation", font=HEADER_FONT, fill="#0f172a")
    notes = [
        "Symbolic planning state = a set of true facts.",
        "Actions are grounded STRIPS operators with preconditions and add/delete effects.",
        "A planner searches over these symbolic states until every goal fact is true.",
        "The same final plan replay can hide very different search effort.",
    ]
    for idx, note in enumerate(notes):
        draw.text((900, 862 + idx * 25), note, font=BODY_FONT, fill="#111827")

    image.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render expanded/generated state exploration for symbolic planner methods.")
    parser.add_argument("--env", action="append", default=[], choices=rv.ENVIRONMENTS, help="Environment file. Can be repeated.")
    parser.add_argument("--method", action="append", default=[], choices=sorted(rv.METHODS), help="Method to visualize. Can be repeated.")
    parser.add_argument("--max-expansions", type=int, default=100000)
    parser.add_argument("--max-frames", type=int, default=36)
    parser.add_argument("--comparison-only", action="store_true", help="Only render the side-by-side method comparison for each environment.")
    args = parser.parse_args()

    envs = args.env or ["Blocks.txt", "BlocksTriangle.txt", "FireExtinguisher.txt"]
    methods = args.method or ["bfs", "astar_goal", "strong"]

    for env_name in envs:
        env = rv.parse_environment(rv.ENVS_DIR / env_name)
        env_results: dict[str, SearchResult] = {}
        for method in methods:
            result = simulate_search(env, method, args.max_expansions)
            env_results[method] = result
            if not args.comparison_only:
                panel, gif = render_search_exploration(env, method, result, args.max_frames)
                print(
                    f"{env_name:28s} {method:12s} solved={result.solved} "
                    f"expanded={len(result.expanded_order)} generated={len(result.nodes) - 1} "
                    f"panel={panel} gif={gif}"
                )
            else:
                print(
                    f"{env_name:28s} {method:12s} solved={result.solved} "
                    f"expanded={len(result.expanded_order)} generated={len(result.nodes) - 1}"
                )
        if len(env_results) > 1:
            panel, gif = render_method_comparison(env, env_results, max_frames=min(args.max_frames, 24))
            print(f"{env_name:28s} comparison   panel={panel} gif={gif}")
            dashboard = render_search_dashboard(env, env_results)
            print(f"{env_name:28s} dashboard    panel={dashboard}")


if __name__ == "__main__":
    main()
