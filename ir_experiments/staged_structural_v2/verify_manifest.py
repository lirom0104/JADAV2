#!/usr/bin/env python3
"""Read-only hash guard for this staged patch. Does not import torch or apply it."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def verify(root_state="before"):
    stage = Path(__file__).resolve().parent
    manifest = json.loads((stage / "manifest.json").read_text())
    root = Path(manifest["repo_root"])
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for version, key in (("before", "base_sha256"), ("after", "staged_sha256")):
        for name, expected in manifest[key].items():
            if sha(stage / version / name) != expected:
                raise ValueError(f"Staged {version} hash differs: {name}")
    patch = stage / manifest["patch_file"]
    if sha(patch) != manifest["patch_sha256"]:
        raise ValueError("Patch hash differs")
    regenerated = "".join("".join(difflib.unified_diff(
        (stage / "before" / name).read_text().splitlines(True),
        (stage / "after" / name).read_text().splitlines(True),
        fromfile="a/" + name, tofile="b/" + name)) for name in manifest["changed_files"])
    if patch.read_text() != regenerated:
        raise ValueError("Patch is not exactly reproducible from before/after copies")
    for name, expected in manifest["protected_root_sha256"].items():
        if root_state == "after" and name in manifest["changed_files"]:
            expected = manifest["staged_sha256"][name]
        if sha(root / name) != expected:
            raise ValueError(f"Root hash differs from requested {root_state} state: {name}")
    frozen = Path(manifest["protected_frozen_source_root"])
    for name, expected in manifest["protected_frozen_sha256"].items():
        if sha(frozen / name) != expected:
            raise ValueError(f"Frozen v1 source hash differs: {name}")
    return dict(verified=True, root_state=root_state, changed_files=manifest["changed_files"],
                patch_sha256=manifest["patch_sha256"], staged_source_directory=str(stage / "after"),
                frozen_v1_checked_files=len(manifest["protected_frozen_sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-state", choices=("before", "after"), default="before")
    print(json.dumps(verify(parser.parse_args().root_state), indent=2, sort_keys=True))
