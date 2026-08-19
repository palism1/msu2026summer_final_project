# FILE MAP | Ensemble / cascade factory exports. Imports are LAZY (PEP 562 __getattr__), same
#   shape as src/models/__init__.py, so a torch-free test can pull in `combine` or `cache` helpers
#   without dragging torch / segment-anything along. `cascade.py` stays torch-free at module level
#   too (see its own FILE MAP), so importing it here is also cheap.

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    "cache_dir": "cache",
    "cache_path": "cache",
    "pack_gt": "cache",
    "unpack_gt": "cache",
    "gt_fingerprint": "cache",
    "SplitCache": "cache",
    "write_split_cache": "cache",
    "read_split_cache": "cache",
    "write_manifest": "cache",
    "read_manifest": "cache",
    "assert_aligned": "cache",
    "environment_record": "cache",
    "normalize_weights": "combine",
    "average_probs": "combine",
    "evaluate_stack": "combine",
    "calibration_stats": "combine",
    "merge_backbones": "combine",
    "merge_normalization": "combine",
    "fit_holdout_indices": "combine",
    "simplex_grid": "combine",
    "WeightFit": "combine",
    "fit_weights": "combine",
    "build_fit_stack": "combine",
    "largest_component": "cascade",
    "box_from_prediction": "cascade",
    "load_image_uint8": "cascade",
    "CascadeSplitResult": "cascade",
    "run_cascade_split": "cascade",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
