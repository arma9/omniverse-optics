"""
Path helper utilities for locating shared resources inside the project.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence


@lru_cache(maxsize=1)
def find_project_root() -> Path:
    """Return the repository root (detected via repo.toml or .git)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "repo.toml").exists() or (parent / ".git").is_dir():
            return parent
    # Fallback to the highest directory available
    return current.parents[-1]


@lru_cache(maxsize=1)
def get_extension_root() -> Path:
    """Return the extension root directory (mit.test_extensions_1)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "mit.test_extensions_1":
            return parent
    # Default to the canonical location under source/extensions
    return find_project_root() / "source" / "extensions" / "mit.test_extensions_1"


def get_extension_data_dir() -> Path:
    """Return the data directory inside the extension, creating it if needed."""
    data_dir = get_extension_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_graphs_dir() -> Path:
    """Ensure the local graphs data directory exists and return it."""
    graphs_dir = get_extension_data_dir() / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


def get_candidate_graph_dirs() -> List[Path]:
    """List directories that may contain graph JSON files."""
    candidates: List[Path] = []
    chromatix_dir = find_project_root() / "source" / "physics" / "chromatix"
    if chromatix_dir.exists():
        candidates.append(chromatix_dir)

    extension_root = get_extension_root()
    candidates.append(extension_root)

    graphs_dir = ensure_graphs_dir()
    candidates.append(graphs_dir)

    # Include the top-level extension folder (contains example_4f_system.json)
    if extension_root.parent.exists():
        candidates.append(extension_root.parent)

    return candidates


def find_graph_file(preferred_names: Sequence[str]) -> Path | None:
    """Return the first existing graph file matching any of the preferred names."""
    for directory in get_candidate_graph_dirs():
        for name in preferred_names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def get_prefabs_dir() -> Path:
    """Return the CAD prefabs directory."""
    prefabs_dir = find_project_root() / "CAD" / "prefabs"
    prefabs_dir.mkdir(parents=True, exist_ok=True)
    return prefabs_dir


def get_simulation_output_dir(subdir: str = "simulation_results") -> Path:
    """Return a directory for simulation output under the project root."""
    output_dir = find_project_root() / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_chromatix_src_dir() -> Path | None:
    """Return the chromatix source directory if it exists."""
    src_dir = find_project_root() / "source" / "physics" / "chromatix" / "src"
    if src_dir.exists():
        return src_dir
    return None


def get_chromatix_docs_dir() -> Path | None:
    """Return the chromatix docs/experiments directory for external editors."""
    docs_dir = find_project_root() / "source" / "physics" / "chromatix" / "docs" / "experiments"
    if docs_dir.exists():
        return docs_dir
    return None


def get_existing_paths(paths: Iterable[Path]) -> List[Path]:
    """Filter and return paths that currently exist."""
    return [path for path in paths if path and path.exists()]
