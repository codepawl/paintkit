"""Command line: repaint a picture in watercolour, or film it being painted.

    paintkit paint photo.jpg -o painting.png [--size 2048] [--seed 7]
    paintkit film photo.jpg -o film.mp4 [--size 1024] [--seconds 22]

The picture only guides where strokes go and what colour they carry; every output pixel is laid by a brush
stroke on simulated paper. `film` needs ffmpeg on the PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import colour as colour_module
from . import glazing
from . import painterly

PAINTERS = {"classic": painterly, "glazing": glazing}

DESIGN_HEIGHT = 4096  # painterly's brush sizes are given for a picture this tall
WET_DARKENING = 1.15  # fresh paint reads darker than the dried wash
FPS = 30
HOLD_SECONDS = 2.5
PAINTER_HELP = "classic: fast, loose washes. glazing: slower, closer to the picture, deeper darks"


def load_target(path: Path, size: int) -> np.ndarray:
    """sRGB float picture with its long side at `size` pixels (both sides even, for video)."""
    image = Image.open(path).convert("RGB")
    ratio = size / max(image.width, image.height)
    width = max(2, int(round(image.width * ratio / 2)) * 2)
    height = max(2, int(round(image.height * ratio / 2)) * 2)
    image = image.resize((width, height), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def paint_picture(target: np.ndarray, seed: int, recorder=None, log=None, painter: str = "classic"):
    height, width = target.shape[:2]
    scale = height / DESIGN_HEIGHT
    fallback_angle = np.zeros((height, width), dtype=np.float32)
    fallback_weight = np.full((height, width), 0.12, dtype=np.float32)
    return PAINTERS[painter].paint(target, scale, fallback_angle, fallback_weight, None, seed=seed,
                                 log=log or (lambda message: None), recorder=recorder)


def to_bytes(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def command_paint(arguments) -> None:
    started = time.perf_counter()
    target = load_target(arguments.image, arguments.size)
    paper = paint_picture(target, arguments.seed, log=print if arguments.verbose else None,
                          painter=arguments.painter)
    Image.fromarray(to_bytes(paper.render())).save(arguments.output)
    print("painted %s (%dx%d) in %.0f s" % (arguments.output, target.shape[1], target.shape[0],
                                            time.perf_counter() - started))


def command_film(arguments) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("paintkit film needs ffmpeg on the PATH")
    started = time.perf_counter()
    target = load_target(arguments.image, arguments.size)
    height, width = target.shape[:2]
    snapshots = []

    def record(paper, wet_pigment) -> None:
        dried = paper.density
        if wet_pigment is not None:
            paper.density = dried + wet_pigment * WET_DARKENING
        snapshots.append(to_bytes(paper.render()))
        paper.density = dried

    paper = paint_picture(target, arguments.seed, recorder=record, painter=arguments.painter)
    snapshots.append(to_bytes(paper.render()))
    if arguments.still:
        Image.fromarray(snapshots[-1]).save(arguments.still)

    # each snapshot gets screen time by how much it changes the picture, so late small touches pass quickly
    blank = np.full_like(snapshots[0], 255)
    paper_linear = np.array(colour_module.parse(paper.colour)[:3], dtype=np.float32)
    blank[:] = to_bytes(colour_module.linear_to_srgb(paper_linear))
    changes = []
    previous = blank.astype(np.int16)
    for snapshot in snapshots:
        changes.append(float(np.abs(snapshot.astype(np.int16) - previous).mean()) ** 0.5)
        previous = snapshot.astype(np.int16)
    weights = np.asarray(changes) / max(sum(changes), 1e-6)
    frame_counts = np.clip(np.round(weights * arguments.seconds * FPS), 1, 8).astype(int)

    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (width, height),
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(arguments.output)], stdin=subprocess.PIPE)
    shown = blank.astype(np.float32)
    for snapshot, frame_count in zip(snapshots, frame_counts):
        current = snapshot.astype(np.float32)
        for step in range(1, frame_count + 1):
            mix = step / frame_count
            encoder.stdin.write((shown * (1 - mix) + current * mix + 0.5).astype(np.uint8).tobytes())
        shown = current
    for _ in range(int(HOLD_SECONDS * FPS)):
        encoder.stdin.write(snapshots[-1].tobytes())
    encoder.stdin.close()
    encoder.wait()
    seconds = int(frame_counts.sum()) / FPS + HOLD_SECONDS
    print("filmed %s (%dx%d, %.1f s, %d snapshots) in %.0f s" % (arguments.output, width, height, seconds,
                                                                len(snapshots), time.perf_counter() - started))


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="paintkit", description="Repaint a picture in watercolour with code brushes.")
    commands = parser.add_subparsers(dest="command", required=True)

    paint = commands.add_parser("paint", help="repaint a picture as a watercolour PNG")
    paint.add_argument("image", type=Path)
    paint.add_argument("-o", "--output", type=Path, required=True)
    paint.add_argument("--size", type=int, default=2048, help="long side of the painting in pixels (default 2048)")
    paint.add_argument("--seed", type=int, default=7)
    paint.add_argument("--verbose", action="store_true", help="print each pass")
    paint.add_argument("--painter", choices=sorted(PAINTERS), default="classic", help=PAINTER_HELP)
    paint.set_defaults(handler=command_paint)

    film = commands.add_parser("film", help="film the painting stroke by stroke as an MP4")
    film.add_argument("image", type=Path)
    film.add_argument("-o", "--output", type=Path, required=True)
    film.add_argument("--size", type=int, default=1024, help="long side of the film in pixels (default 1024)")
    film.add_argument("--seconds", type=float, default=22.0, help="time spent painting before the final hold")
    film.add_argument("--seed", type=int, default=7)
    film.add_argument("--still", type=Path, help="also save the finished painting as a PNG")
    film.add_argument("--painter", choices=sorted(PAINTERS), default="classic", help=PAINTER_HELP)
    film.set_defaults(handler=command_film)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
