from __future__ import annotations

import importlib.util
from pathlib import Path

_pipeline_py = Path(__file__).resolve().parents[1] / "pipeline.py"
_spec = importlib.util.spec_from_file_location("_truerender_pipeline_py", _pipeline_py)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_pipeline_py}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

load_models = _module.load_models
segment_and_crop = _module.segment_and_crop
generate_mesh = _module.generate_mesh
convert_obj_to_glb = _module.convert_obj_to_glb

__all__ = [
    "load_models",
    "segment_and_crop",
    "generate_mesh",
    "convert_obj_to_glb",
]
