"""Create a verified source/model archive without local environments or Git internals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    ".build",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}


def project_files():
    for current, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.endswith(".egg-info"))
        for name in sorted(names):
            path = Path(current) / name
            relative = path.relative_to(ROOT)
            if path.is_symlink() or name.endswith((".zip", ".zip.tmp", ".sha256.txt")):
                continue
            if relative.as_posix().startswith("results/validation/initial_smoke/"):
                continue
            if name == "MANIFEST_SHA256.json":
                continue
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables" / "DEEP_RL_ameliore.zip")
    args = parser.parse_args(argv)
    target = args.output.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Archive must be written inside the project")
    if target.exists():
        raise FileExistsError("Choose a new archive output path")
    paths = list(project_files())
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = ROOT / "deliverables" / "MANIFEST_SHA256.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths + [manifest]:
            archive.write(path, "DEEP_REINFORCEMENT_LEARNING/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for name, expected in hashes.items():
            content = archive.read("DEEP_REINFORCEMENT_LEARNING/" + name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError("Archive SHA-256 verification failed: " + name)
    temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256.txt").write_text(
        digest + "  " + target.name + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "archive": str(target),
                "files": len(paths) + 1,
                "bytes": target.stat().st_size,
                "sha256": digest,
                "crc_and_content_hashes_verified": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
