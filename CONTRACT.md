# paintkit contract (shared by every module)

paintkit is a raster painting toolkit in Python (numpy, scipy, scikit-image, pillow only) that a model
drives from code to paint illustrations the way an artist uses Krita or Procreate.

Run everything with `uv run --no-project --with numpy --with scipy --with scikit-image --with pillow python <file>`
from the repository root.

## Pixel format

- A layer's pixels are `numpy.float32` arrays of shape `(height, width, 4)`, **premultiplied** RGBA,
  colour channels in **linear light**, values 0..1. Convert sRGB in and out only at the edges
  (`colour.srgb_to_linear`, `colour.linear_to_srgb`).
- A colour argument anywhere in the API is accepted as `"#rrggbb"`, `"#rrggbbaa"`, an `(r, g, b)` or
  `(r, g, b, a)` tuple of sRGB floats 0..1; `colour.parse(value)` returns a linear `(r, g, b, a)` tuple.
- A selection or mask is a `numpy.float32` array `(height, width)` 0..1 (1 = fully selected). `None`
  means everything is selected.
- Coordinates: x right, y down, pixels, floats allowed. Angles in degrees.

## Shared objects (in `paintkit/core.py`)

```python
class Layer:
    name: str
    pixels: np.ndarray        # (H, W, 4) float32 premultiplied linear
    opacity: float = 1.0
    blend: str = "normal"
    visible: bool = True
    alpha_lock: bool = False  # painting keeps existing alpha
    clip_to_below: bool = False
    mask: np.ndarray | None   # (H, W) layer mask

class Canvas:
    width: int; height: int
    layers: list[Layer]
    selection: np.ndarray | None
    def new_layer(self, name, blend="normal", opacity=1.0) -> Layer
    def composite(self) -> np.ndarray   # (H, W, 4) premultiplied linear
    def save_png(self, path)            # sRGB 8-bit
    def save_ora(self, path)            # OpenRaster, opens in Krita with layers
```

## How a brush paints (the seam between core and brushes)

Every painting operation ends in `core.apply_dab(layer, dab_rgba, x0, y0, selection, mode="normal")`:
`dab_rgba` is a small premultiplied `(h, w, 4)` array placed with its top-left at integer `(x0, y0)`;
it is clipped to the canvas, multiplied by the selection, composited onto `layer.pixels` with `mode`
(`normal`, `erase`, `multiply`, `screen`, `add`, `lighten`, `darken`) and respects `layer.alpha_lock`.
Brushes build dabs; they never write `layer.pixels` directly except smudge/blur/wet tools, which read
the area under the dab first through `core.read_area(layer, x0, y0, w, h)`.

A stroke is a list of `StrokePoint(x, y, pressure=1.0, tilt=0.0, time=None)`; `paths.py` produces
these from lines, curves and shapes, with pressure shaped by taper settings.
