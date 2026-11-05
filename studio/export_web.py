"""Bridges from the Python studio to the browser studio.

    python -m studio.export_web
        writes studio/web/data/catalog.json, the fixed data the browser version reads (every setting's label,
        help, range and default, the world templates and the presets). Run it after changing any of those.
    python -m studio.export_web walker <training id, training folder or Python .walker file> [more ...]
        converts walkers trained in the Python studio into browser .walker files (brains included), saved
        next to the source (or in the current folder for a training id). Open them with "Open a shared walker".
"""
import base64
import gzip
import json
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from studio.server import catalog

DATA = Path(__file__).parent / "web" / "data"
WALKER_FORMAT = "walk-studio-walker"  # the browser's .walker format (js/engine/sharing.js)


# ---------- Brains ----------

def _layer(linear) -> dict:
    weight = linear.weight.detach().cpu().numpy()
    return {"w": [float(v) for v in weight.ravel()], "b": [float(v) for v in linear.bias.detach().cpu().numpy()],
            "inputs": int(weight.shape[1]), "outputs": int(weight.shape[0])}


def _linears(sequence) -> list[dict]:
    import torch
    return [_layer(module) for module in sequence if isinstance(module, torch.nn.Linear)]


# A Stable-Baselines3 policy (free or symmetric) as the plain brain data of the browser (see js/engine/network.js)
def brain_to_web(policy, activation: str, mirror: dict | None) -> dict:
    extractor = policy.mlp_extractor
    if hasattr(extractor, "actor_trunk"):  # the symmetric brain
        return {"kind": "symmetric", "activation": activation, "mirror": mirror,
                "actor": {"layers": _linears(extractor.actor_trunk), "head": _layer(extractor.actor_head)},
                "critic": {"layers": _linears(extractor.critic_trunk), "head": _layer(extractor.critic_head)},
                "logStd": [float(v) for v in policy.log_std.detach().cpu().numpy()]}
    return {"kind": "free", "activation": activation,
            "actor": {"layers": _linears(extractor.policy_net), "head": _layer(policy.action_net)},
            "critic": {"layers": _linears(extractor.value_net), "head": _layer(policy.value_net)},
            "logStd": [float(v) for v in policy.log_std.detach().cpu().numpy()]}


def export_catalog() -> Path:
    data = catalog()
    data.pop("system", None)  # facts about this computer: the browser measures its own
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "catalog.json"
    target.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


# ---------- Walkers for the browser ----------

# A list of numbers as the browser's base64 Float32Array ({"$f32": ...}, little endian)
def _f32(values) -> dict:
    return {"$f32": base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")}


def _packed_brain(brain: dict) -> dict:
    def layer(item):
        return {**item, "w": _f32(item["w"]), "b": _f32(item["b"])}

    def net(item):
        return {"layers": [layer(l) for l in item["layers"]], "head": layer(item["head"])}

    return {**brain, "actor": net(brain["actor"]), "critic": net(brain["critic"]), "logStd": _f32(brain["logStd"])}


# One saved brain of a Python training (brain/<which>.zip and its scaling) in the browser's form
def _brain_file(folder: Path, which: str, setup: dict) -> dict | None:
    model_path, scaling_path = folder / "brain" / f"{which}.zip", folder / "brain" / f"{which}_scaling.pkl"
    if not model_path.exists() or not scaling_path.exists():
        return None
    from stable_baselines3 import PPO
    from studio.brain import clean_design, design_for_body, design_senses
    from studio.creatures import build_creature
    from studio.mirror import mirror_tables
    model = PPO.load(str(model_path), device="cpu")
    with scaling_path.open("rb") as handle:
        scaling = pickle.load(handle)  # a VecNormalize without its worlds: obs_rms, ret_rms and clip_obs
    build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
    design = design_for_body(clean_design(setup.get("brain")), build)
    mirror = mirror_tables(build, design_senses(design)).to_dict() if design["values"]["symmetric"] else None
    brain = brain_to_web(model.policy, design["values"]["activation"], mirror)
    brain["scaling"] = {"mean": _f32(scaling.obs_rms.mean), "var": _f32(scaling.obs_rms.var), "clip": float(scaling.clip_obs)}
    rms = lambda r: {"mean": [float(v) for v in np.ravel(r.mean)], "var": [float(v) for v in np.ravel(r.var)], "count": float(r.count)}
    rate = model.learning_rate if isinstance(model.learning_rate, (int, float)) else 3e-4
    return {"brain": _packed_brain(brain), "obs_rms": rms(scaling.obs_rms), "ret_rms": rms(scaling.ret_rms),
            "learning_rate": float(rate), "steps": int(model.num_timesteps)}


# A Python training folder as the browser's .walker file (gzip JSON)
def walker_from_folder(folder: Path, target: Path) -> Path:
    from studio import library
    from studio.manager import read_rows
    setup = library.read_json(folder / "setup.json")
    if not setup:
        raise ValueError(f"{folder} is not a training folder (it has no setup.json).")
    thumbnail = folder / "thumbnail.jpg"
    replays = {path.stem: library.read_json(path) for path in sorted((folder / "replays").glob("*.json"))}
    data = {
        "format": WALKER_FORMAT, "version": 1, "setup": setup, "status": library.read_json(folder / "status.json", {}) or {},
        "thumbnail": "data:image/jpeg;base64," + base64.b64encode(thumbnail.read_bytes()).decode("ascii") if thumbnail.exists() else None,
        "progress": read_rows(folder / "progress.csv"), "exams": read_rows(folder / "exams.csv"),
        "challenges": library.read_json(folder / "challenges.json"), "replays": replays,
        "brains": {which: brain for which in ("best", "latest") if (brain := _brain_file(folder, which, setup))},
    }
    if not data["brains"]:
        raise ValueError(f"{setup.get('name', folder.name)} has no trained brain yet.")
    target.write_bytes(gzip.compress(json.dumps(data).encode("utf-8")))
    return target


# A training id, a training folder or a Python .walker (zip) file, converted for the browser
def convert_walker(source: str) -> Path:
    from studio import library
    path = Path(source)
    if path.is_file() and zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():  # only the walker's own files, never outside the folder
                    if ".." not in Path(name).parts and not name.startswith("/"):
                        archive.extract(name, folder)
            return walker_from_folder(folder, path.with_name(f"{path.stem} (browser).walker"))
    folder = path if path.is_dir() else library.training_folder(source)
    setup = library.read_json(folder / "setup.json") or {}
    name = "".join(c if c.isalnum() or c in " -_.()" else "_" for c in setup.get("name", folder.name)).strip() or "walker"
    target = (folder.parent if path.is_dir() else Path.cwd()) / f"{name}.walker"
    return walker_from_folder(folder, target)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "walker":
        for source in sys.argv[2:]:
            print(f"Wrote {convert_walker(source)}")
    else:
        path = export_catalog()
        print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
