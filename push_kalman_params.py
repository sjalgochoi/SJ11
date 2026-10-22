"""Build parameter-index-0 JSON and push it to the SJ11 GitHub repository.

Usage from the repository root:
    python push_kalman_params.py
    python push_kalman_params.py --source-dir ../2factor_vol_ma3_1500_obs9_111

Requires Git configured with write access to sjalgochoi/SJ11.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PARAMETER_INDEX = 0
REPOSITORY_URL = "https://github.com/sjalgochoi/SJ11.git"


def find_git() -> str:
    """Find Git on PATH or in the Codex Windows runtime installation."""
    git_on_path = shutil.which("git")
    if git_on_path:
        return git_on_path

    home = Path.home()
    candidates = [
        home
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe",
        Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))
        / "Programs/Git/cmd/git.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Git/cmd/git.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Git/cmd/git.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Git was not found. Install Git for Windows, add git.exe to PATH, or set "
        "GIT_EXE to the full path of git.exe."
    )


def read_matrix(path: Path) -> list[list[float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = [[cell.strip() for cell in row] for row in csv.reader(source)]
    rows = [row for row in rows if row and any(row)]
    if not rows:
        raise ValueError(f"Empty parameter file: {path}")

    # The BO CSVs use a pandas column header: 0, 1, ..., N-1.
    try:
        header = [int(value) for value in rows[0]]
    except ValueError:
        header = []
    if header == list(range(len(header))):
        rows = rows[1:]

    try:
        matrix = [[float(value) for value in row] for row in rows]
    except ValueError as error:
        raise ValueError(f"Non-numeric value in {path}") from error
    if not matrix or len({len(row) for row in matrix}) != 1:
        raise ValueError(f"Empty or inconsistent matrix dimensions in {path}")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"Non-finite value in {path}; JSON cannot represent it")
    return matrix


def build_document(source_dir: Path) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Parameter source directory not found: {source_dir}")
    train_sizes: dict[str, Any] = {}
    for train_dir in sorted(
        (path for path in source_dir.iterdir() if path.is_dir()),
        key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name),
    ):
        parameter_files = sorted(train_dir.glob(f"*_{PARAMETER_INDEX}.csv"))
        if not parameter_files:
            continue
        train_sizes[train_dir.name] = {
            path.stem.rsplit("_", 1)[0]: read_matrix(path)
            for path in parameter_files
        }
    if not train_sizes:
        raise FileNotFoundError(
            f"No *_{PARAMETER_INDEX}.csv files found under {source_dir}"
        )
    return {
        "parameter_index": PARAMETER_INDEX,
        "source_directory": source_dir.name,
        "train_sizes": train_sizes,
    }


def run_git(repo_dir: Path, *args: str) -> str:
    completed = subprocess.run(
        [find_git(), "-C", str(repo_dir), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    # Preserve leading spaces in `git status --porcelain` records.
    return completed.stdout.rstrip("\r\n")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_dir.parent / "2factor_vol_ma3_1500_obs9_111",
        help="Directory containing the train-size folders and parameter CSVs",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()

    if run_git(repo_dir, "remote", "get-url", "origin").rstrip("/").removesuffix(".git") != REPOSITORY_URL.removesuffix(".git"):
        raise RuntimeError(f"This checkout does not have expected origin {REPOSITORY_URL}")
    branch = run_git(repo_dir, "branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"Expected branch 'main', found {branch!r}")
    status_lines = run_git(repo_dir, "status", "--porcelain").splitlines()
    allowed_changes = {"kalman_param.jason", "parameters_0.json", Path(__file__).name}
    unexpected_changes = [line for line in status_lines if line[3:] not in allowed_changes]
    if unexpected_changes:
        raise RuntimeError(
            "Working tree has unrelated changes; commit or stash them before running"
        )
    git_name = run_git(repo_dir, "config", "user.name")
    git_email = run_git(repo_dir, "config", "user.email")
    if not git_name or not git_email:
        raise RuntimeError(
            "Git commit identity is not configured. Set user.name and user.email "
            "for this checkout, then rerun."
        )

    document = build_document(source_dir)
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    (repo_dir / "parameters_0.json").write_text(serialized, encoding="utf-8")
    # Keep the repository's established filename, including its .jason spelling.
    (repo_dir / "kalman_param.jason").write_text(serialized, encoding="utf-8")

    run_git(repo_dir, "add", "parameters_0.json", "kalman_param.jason", Path(__file__).name)
    if not run_git(repo_dir, "status", "--porcelain"):
        print("The JSON files are already up to date; nothing to commit or push.")
        return
    run_git(repo_dir, "commit", "Update Kalman parameters for index 0")
    run_git(repo_dir, "push", "origin", "main")
    print(
        f"Pushed index {PARAMETER_INDEX} parameters for "
        f"{len(document['train_sizes'])} training sizes to {REPOSITORY_URL}"
    )


if __name__ == "__main__":
    main()
