"""The command line paints a small picture end to end, and the film recorder keeps the result."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paintkit import cli  # noqa: E402


def make_picture(path: Path) -> None:
    ys, xs = np.mgrid[0:240, 0:180]
    picture = np.zeros((240, 180, 3), dtype=np.float32)
    picture[..., 0] = 0.2 + 0.6 * (xs / 180)
    picture[..., 1] = 0.3
    picture[..., 2] = 0.8 - 0.5 * (ys / 240)
    disc = (xs - 90) ** 2 + (ys - 100) ** 2 < 40 ** 2
    picture[disc] = (0.95, 0.8, 0.2)
    Image.fromarray((picture * 255).astype(np.uint8)).save(path)


def test_paint_writes_a_painting_that_follows_the_picture(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "painting.png"
    make_picture(source)
    cli.main(["paint", str(source), "-o", str(output), "--size", "240"])
    painting = np.asarray(Image.open(output)).astype(np.float32) / 255
    assert painting.shape == (240, 180, 3)
    disc = painting[90:110, 80:100].mean(axis=(0, 1))
    corner = painting[200:230, 10:40].mean(axis=(0, 1))
    assert disc[0] > disc[2], "the yellow disc should read warm"
    assert corner[2] > corner[0], "the lower left should read blue"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_film_writes_a_video_and_the_final_still(tmp_path):
    source = tmp_path / "source.png"
    film = tmp_path / "film.mp4"
    still = tmp_path / "still.png"
    make_picture(source)
    cli.main(["film", str(source), "-o", str(film), "--size", "160", "--seconds", "2", "--still", str(still)])
    assert film.stat().st_size > 10000
    assert np.asarray(Image.open(still)).shape[:2] == (160, 120)


def test_glazing_painter_follows_the_picture_and_keeps_darks(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "painting.png"
    make_picture(source)
    cli.main(["paint", str(source), "-o", str(output), "--size", "240", "--painter", "glazing"])
    painting = np.asarray(Image.open(output)).astype(np.float32) / 255
    reference = np.asarray(Image.open(source)).astype(np.float32) / 255
    assert painting.shape == (240, 180, 3)
    disc = painting[90:110, 80:100].mean(axis=(0, 1))
    corner = painting[200:230, 10:40].mean(axis=(0, 1))
    assert disc[0] > disc[2], "the yellow disc should read warm"
    assert corner[2] > corner[0], "the lower left should read blue"
    assert np.abs(painting - reference).mean() < 0.12, "the glazing painter should stay close to the picture"
