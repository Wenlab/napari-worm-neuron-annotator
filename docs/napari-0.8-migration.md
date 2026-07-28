# napari 0.8 compatibility migration

This plugin targets napari `>=0.8,<0.9` and Python `>=3.11,<3.15`.

## Dependency policy

- The base package depends on `napari`, `numpy`, and `qtpy`.
- It does not install PyQt or PySide. The host napari environment chooses its
  Qt backend.
- The `all` extra delegates the fresh-environment Qt choice to
  `napari[all]`.
- The Pixi development environment uses PyQt6 only for local testing.
- Excel support remains isolated in the `excel` extra and Pixi environment.

## API migration audit

- All Qt imports use `qtpy`; there are no imports from PyQt5, PyQt6, or
  PySide.
- Label RGB values are read from the active napari colormap. The plugin does
  not use the `color_dict_to_colormap` helper removed in napari 0.8.
- Derived box overlays use the public `Viewer.add_vectors`, `Vectors.data`,
  `Vectors.features`, and dims event APIs.
- Managed layer identity is stored in public layer metadata rather than in
  private napari objects or user-editable layer names.
- 3D and 4D Labels switching recreates Vectors with the correct dimensionality
  and does not retain duplicate managed layers.

## Verification baseline

The migration was tested on:

```text
Python 3.13.14
napari 0.8.0
PyQt6 6.8.1
```

Run the core and optional Excel suites independently:

```text
pixi run pytest -q
pixi run -e excel pytest -q
pixi run ruff check .
```
