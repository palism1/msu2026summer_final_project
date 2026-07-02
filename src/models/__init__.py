# FILE MAP | Model factory exports. Imports are LAZY (PEP 562 __getattr__) so that pulling in
#   a torch-free helper (e.g. box_from_mask for a GPU-free unit test) does not drag torch /
#   segmentation-models / segment-anything along with it. `from src.models import build_unet`
#   still works — the submodule is imported on first attribute access.

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    "build_unet": "unet",
    "build_sam_lora": "sam_adapter",
    "build_medsam_lora": "sam_adapter",
    "build_zeroshot_sam": "zeroshot",
    "build_zeroshot_medsam": "zeroshot",
    "box_from_mask": "zeroshot",
    "point_from_mask": "zeroshot",
    "is_zeroshot": "zeroshot",
    "predict_zeroshot_prob": "zeroshot",
    "evaluate_zeroshot_all_splits": "zeroshot",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
