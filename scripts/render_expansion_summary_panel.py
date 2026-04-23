#!/usr/bin/env python3
"""Render a lightweight 8-panel search-expansion summary PNG."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
VIS_ROOT = PROJECT_ROOT / "outputs" / "visualizations"
OUT_DIR = VIS_ROOT / "summary"
OUT_PATH = OUT_DIR / "expansion_by_method_panel.png"

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

METHOD_LABELS = {
    "bfs": "Breadth First Search",
    "astar": "A* relaxed plan",
    "astar_goal": "A* goal count",
    "astar_hadd": "A* additive relaxation",
    "optimal": "A* max relaxation",
    "strong": "Strong A* additive",
    "weighted_ff": "Weighted A*",
    "greedy_ff": "Greedy best first",
}

METHODS = list(METHOD_LABELS)

ENV_LABELS = {
    "BlocksEasy.txt": "BlocksEasy",
    "Blocks.txt": "Blocks",
    "BlocksTower4.txt": "Tower4",
    "BlocksTwoStack5.txt": "TwoStack5",
    "BlocksSixPairing.txt": "SixPairing",
    "BlocksTriangle.txt": "Triangle",
    "BlocksTriangleBridge.txt": "TriBridge",
    "BlocksTriangleTwinTowers.txt": "Twin Towers",
    "FireExtinguisher.txt": "Fire",
    "FireExtinguisherReturn.txt": "Fire Return",
    "HospitalThreeRobotDelivery.txt": "Hospital",
    "DisasterResponseThreeRobot.txt": "Disaster",
    "WarehouseThreeRobotFulfillment.txt": "Warehouse",
}

COLORS = [
    "#1d4ed8",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#f97316",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#ca8a04",
    "#0f766e",
    "#7c2d12",
    "#334155",
]


def tick_label(value: int) -> str:
    if value >= 1000:
        return f"{value // 1000}k"
    return str(value)


def nice_y_max(value: int) -> int:
    for candidate in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        if value <= candidate:
            return candidate
    return 200000


def y_ticks(y_max: int) -> List[int]:
    ticks = [0, 1, 10, 100, 1000, 10000, 100000]
    visible = [tick for tick in ticks if tick <= y_max]
    if y_max not in visible:
        visible.append(y_max)
    return visible


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def profile_path(env_name: str, method: str) -> Path:
    return RUNS_ROOT / env_name.replace(".txt", "") / method / "search_profile.json"


def fallback_profile(env_name: str, method: str) -> Tuple[List[float], List[int]]:
    run_dir = RUNS_ROOT / env_name.replace(".txt", "") / method
    metrics_path = run_dir / "metrics.json"
    plan_path = run_dir / "plan.txt"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    plan_length = len([line for line in plan_path.read_text().splitlines() if line.strip()]) if plan_path.exists() else 0
    total = int(metrics.get("expanded", 0) or 0)
    denom = max(1, plan_length)
    steps = list(range(plan_length + 1))
    return [step / denom for step in steps], [round(total * step / denom) for step in steps]


def load_profile(env_name: str, method: str) -> Tuple[List[float], List[int]]:
    path = profile_path(env_name, method)
    try:
        profile = json.loads(path.read_text())
        expanded = [int(value) for value in profile.get("expanded_by_step", [])]
        steps = [int(value) for value in profile.get("steps", [])]
        if not steps or not expanded:
            raise ValueError("empty profile")
        denom = max(max(steps), 1)
        return [step / denom for step in steps], expanded
    except Exception:
        return fallback_profile(env_name, method)


def log_y(value: int, y_max: int, top: int, bottom: int) -> int:
    if value <= 0:
        return bottom
    value = max(1, value)
    max_value = max(10, y_max)
    t = math.log10(value) / math.log10(max_value)
    return int(bottom - t * (bottom - top))


def draw_polyline(draw: ImageDraw.ImageDraw, points: List[Tuple[int, int]], color: str, width: int) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = font(34, True)
    subtitle_font = font(18)
    panel_title_font = font(19, True)
    label_font = font(13)
    small_font = font(11)
    legend_font = font(13)

    profiles: Dict[Tuple[str, str], Tuple[List[float], List[int]]] = {}
    method_max: Dict[str, int] = {}
    for method in METHODS:
        local_max = 1
        for env_name in ENVIRONMENTS:
            progress, expanded = load_profile(env_name, method)
            profiles[(env_name, method)] = (progress, expanded)
            local_max = max(local_max, max(expanded or [0]))
        method_max[method] = nice_y_max(local_max)

    width, height = 1400, 1900
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 28), "Search Expansion Over Plan Progress", font=title_font, fill="#0f172a", anchor="ma")
    draw.text(
        (width // 2, 68),
        "13 environments across 8 planner methods. Each panel uses a log y scale for its own method.",
        font=subtitle_font,
        fill="#334155",
        anchor="ma",
    )

    legend_x, legend_y = 55, 112
    legend_col_w, legend_row_h = 260, 27
    for idx, env_name in enumerate(ENVIRONMENTS):
        row, col = divmod(idx, 5)
        lx = legend_x + col * legend_col_w
        ly = legend_y + row * legend_row_h
        draw.line((lx, ly + 9, lx + 34, ly + 9), fill=COLORS[idx], width=5)
        draw.text((lx + 48, ly), ENV_LABELS[env_name], font=legend_font, fill="#0f172a")

    panel_w, panel_h = 610, 340
    gap_x, gap_y = 70, 55
    start_x, start_y = 55, 210
    axis_left_pad, axis_right_pad = 68, 22
    axis_top_pad, axis_bottom_pad = 58, 52

    for idx, method in enumerate(METHODS):
        row, col = divmod(idx, 2)
        x0 = start_x + col * (panel_w + gap_x)
        y0 = start_y + row * (panel_h + gap_y)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill="#ffffff", outline="#d5dde8", width=2)
        draw.text((x0 + 18, y0 + 15), METHOD_LABELS[method], font=panel_title_font, fill="#0f172a")

        px0 = x0 + axis_left_pad
        py0 = y0 + axis_top_pad
        px1 = x1 - axis_right_pad
        py1 = y1 - axis_bottom_pad
        draw.line((px0, py1, px1, py1), fill="#334155", width=2)
        draw.line((px0, py0, px0, py1), fill="#334155", width=2)

        local_y_max = method_max[method]
        for tick in y_ticks(local_y_max):
            ty = log_y(tick, local_y_max, py0, py1)
            draw.line((px0, ty, px1, ty), fill="#e2e8f0", width=1)
            draw.text((x0 + 18, ty - 7), tick_label(tick), font=small_font, fill="#64748b")

        for x_tick, text in [(0.0, "0"), (0.5, "50%"), (1.0, "100%")]:
            tx = int(px0 + x_tick * (px1 - px0))
            draw.line((tx, py1, tx, py1 + 5), fill="#334155", width=1)
            draw.text((tx, py1 + 13), text, font=small_font, fill="#475569", anchor="ma")

        for env_idx, env_name in enumerate(ENVIRONMENTS):
            progress, expanded = profiles[(env_name, method)]
            points = []
            for p, value in zip(progress, expanded):
                x = int(px0 + max(0.0, min(1.0, p)) * (px1 - px0))
                y = log_y(value, local_y_max, py0, py1)
                points.append((x, y))
            draw_polyline(draw, points, COLORS[env_idx], 4 if env_name in {"BlocksTriangleTwinTowers.txt", "HospitalThreeRobotDelivery.txt", "WarehouseThreeRobotFulfillment.txt"} else 3)
            if points:
                ex, ey = points[-1]
                draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill=COLORS[env_idx], outline="#ffffff")

        draw.text((x0 + panel_w // 2, y1 - 25), "normalized plan progress", font=label_font, fill="#334155", anchor="ma")
        draw.text((x0 + 18, y0 + 42), "expanded states", font=label_font, fill="#334155")

    image.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
