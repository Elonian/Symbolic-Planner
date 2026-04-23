#!/usr/bin/env python3
"""Build README-ready combined 3D animation GIFs.

Each output GIF shows the same planner method across four representative
environments:

- HospitalThreeRobotDelivery
- FireExtinguisherReturn
- DisasterResponseThreeRobot
- BlocksTriangleBridge

The script uses the saved PNG frame folders instead of recompressing source
GIFs. That keeps the layout stable and lets the four domains be synchronized by
normalized plan progress.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_ROOT = PROJECT_ROOT / "outputs" / "visualizations"
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUT_DIR = VIS_ROOT / "readme_panels"

METHODS = [
    ("strong", "Best Practical Planner", "Strong A* / hadd delete relaxation"),
    ("optimal", "Shortest-Plan A*", "A* / hmax delete relaxation"),
    ("astar_hadd", "A* hadd", "A* / additive delete relaxation"),
    ("astar", "A* relaxed-plan", "A* / FF-style relaxed-plan heuristic"),
    ("astar_goal", "A* goal-count", "A* / unsatisfied goal count"),
    ("weighted_ff", "Weighted A*", "Weighted A* / relaxed-plan, w=5"),
    ("greedy_ff", "Greedy Best First", "Greedy best-first / relaxed-plan"),
    ("bfs", "Breadth First Search", "BFS / no heuristic baseline"),
]

ENVIRONMENTS = [
    {
        "key": "HospitalThreeRobotDelivery",
        "name": "Hospital Delivery",
        "env_file": "HospitalThreeRobotDelivery.txt",
        "frame_dir": "_robot_task_3d_frames",
    },
    {
        "key": "FireExtinguisherReturn",
        "name": "Fire Return Mission",
        "env_file": "FireExtinguisherReturn.txt",
        "frame_dir": "_fire_3d_frames",
    },
    {
        "key": "DisasterResponseThreeRobot",
        "name": "Disaster Response",
        "env_file": "DisasterResponseThreeRobot.txt",
        "frame_dir": "_robot_task_3d_frames",
    },
    {
        "key": "BlocksTriangleBridge",
        "name": "Triangle Bridge",
        "env_file": "BlocksTriangleBridge.txt",
        "frame_dir": "_blockworld_3d_frames",
    },
]

TILE_W = 420
TILE_H = 236
TILE_HEADER_H = 44
MARGIN = 28
GAP = 20
TOP = 84
BOTTOM = 34
CANVAS_W = 2 * TILE_W + GAP + 2 * MARGIN
CANVAS_H = TOP + 2 * (TILE_HEADER_H + TILE_H) + GAP + BOTTOM
FRAME_COUNT = 28
FRAME_DURATION_MS = 210


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


TITLE_FONT = font(26, True)
SUBTITLE_FONT = font(17)
ENV_FONT = font(17, True)
SMALL_FONT = font(12)


def load_summary() -> Dict[Tuple[str, str], Dict[str, object]]:
    path = PROJECT_ROOT / "outputs" / "summary.json"
    rows = json.loads(path.read_text())
    return {(row["environment"], row["method"]): row for row in rows}


def frame_paths(env_key: str, method: str, frame_dir: str) -> List[Path]:
    directory = VIS_ROOT / env_key / method / frame_dir
    frames = sorted(directory.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {directory}")
    return frames


def fit_image(source: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#ffffff")
    image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def rounded_tile(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd5e1", width=2)


def draw_title(draw: ImageDraw.ImageDraw, group_title: str, method_label: str, frame_index: int) -> None:
    title = f"{group_title} across four symbolic domains"
    draw.text((MARGIN, 18), title, fill="#0f172a", font=TITLE_FONT)
    draw.text((MARGIN, 50), f"{method_label} | synchronized 3D replay | frame {frame_index + 1}/{FRAME_COUNT}", fill="#475569", font=SUBTITLE_FONT)
    if group_title == "Best Practical Planner":
        badge = (CANVAS_W - 190, 18, CANVAS_W - MARGIN, 48)
        draw.rounded_rectangle(badge, radius=8, fill="#dbeafe", outline="#bfdbfe")
        draw.text((CANVAS_W - 178, 25), "best practical", fill="#1e3a8a", font=SMALL_FONT)


def tile_position(index: int) -> Tuple[int, int, int, int]:
    row, col = divmod(index, 2)
    x0 = MARGIN + col * (TILE_W + GAP)
    y0 = TOP + row * (TILE_HEADER_H + TILE_H + GAP)
    return x0, y0, x0 + TILE_W, y0 + TILE_HEADER_H + TILE_H


def draw_env_tile(
    canvas: Image.Image,
    env: Dict[str, str],
    source_path: Path,
    method: str,
    summary: Dict[Tuple[str, str], Dict[str, object]],
    index: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = tile_position(index)
    rounded_tile(draw, (x0, y0, x1, y1))
    metrics = summary.get((env["env_file"], method), {})
    metric_text = f"plan {metrics.get('plan_length', '?')} | expanded {metrics.get('expanded', '?')}"
    draw.text((x0 + 14, y0 + 10), env["name"], fill="#0f172a", font=ENV_FONT)
    draw.text((x1 - 14, y0 + 13), metric_text, fill="#475569", font=SMALL_FONT, anchor="ra")

    with Image.open(source_path) as image:
        fitted = fit_image(image, (TILE_W - 20, TILE_H - 18))
    canvas.paste(fitted, (x0 + 10, y0 + TILE_HEADER_H + 8))


def source_for_progress(paths: List[Path], frame_index: int) -> Path:
    if len(paths) == 1:
        return paths[0]
    progress = frame_index / max(1, FRAME_COUNT - 1)
    source_index = round(progress * (len(paths) - 1))
    return paths[source_index]


def build_method_gif(method: str, group_title: str, method_label: str, summary: Dict[Tuple[str, str], Dict[str, object]]) -> Path:
    frame_sets = {
        env["key"]: frame_paths(env["key"], method, env["frame_dir"])
        for env in ENVIRONMENTS
    }

    frames: List[Image.Image] = []
    for frame_index in range(FRAME_COUNT):
        if frame_index == 0 or (frame_index + 1) % 7 == 0:
            print(f"  {method}: composing frame {frame_index + 1}/{FRAME_COUNT}", flush=True)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        draw_title(draw, group_title, method_label, frame_index)
        for env_index, env in enumerate(ENVIRONMENTS):
            path = source_for_progress(frame_sets[env["key"]], frame_index)
            draw_env_tile(canvas, env, path, method, summary, env_index)
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{method}_four_environment_3d.gif"
    print(f"  {method}: saving gif", flush=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined README GIF panels.")
    parser.add_argument("--method", choices=[method for method, _, _ in METHODS], action="append")
    args = parser.parse_args()
    selected = set(args.method or [])

    summary = load_summary()
    outputs = []
    for method, group_title, method_label in METHODS:
        if selected and method not in selected:
            continue
        print(f"building {method}", flush=True)
        out_path = build_method_gif(method, group_title, method_label, summary)
        outputs.append(out_path)
        print(out_path, flush=True)


if __name__ == "__main__":
    main()
