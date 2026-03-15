# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

DEFAULT_PROMPTS_PATH = "config/prompts"


def resolve_prompt_root(raw_path: str, *, base_config_path: str) -> Path:
    """Resolve a prompt root relative to the config file when needed."""

    raw2 = os.path.expanduser(os.path.expandvars(raw_path))
    if os.path.isabs(raw2):
        return Path(raw2)
    base_dir = Path(os.path.abspath(base_config_path)).parent
    return (base_dir / raw2).resolve()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested mappings while letting override win on conflicts."""

    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must contain a YAML mapping: {path}")
    return data


def _normalize_includes(raw: Any, *, path: Path) -> List[str]:
    if raw in (None, "", []):
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"'includes' entries must be non-empty strings in prompt file: {path}"
                )
            out.append(item)
        return out
    raise ValueError(f"'include(s)' must be a string or list in prompt file: {path}")


def resolve_prompt_file(path: Path, *, visited: Iterable[Path] | None = None) -> Dict[str, Any]:
    """Load one prompt file and recursively inline any declared includes."""

    resolved = path.resolve()
    seen = set(visited or [])
    if resolved in seen:
        cycle = " -> ".join(str(p) for p in [*seen, resolved])
        raise ValueError(f"Circular prompt include detected: {cycle}")

    seen.add(resolved)
    data = _read_yaml_mapping(resolved)
    raw_includes = data.pop("includes", data.pop("include", []))
    includes = _normalize_includes(raw_includes, path=resolved)

    merged: Dict[str, Any] = {}
    for rel_include in includes:
        include_path = (resolved.parent / rel_include).resolve()
        merged = deep_merge(
            merged,
            resolve_prompt_file(include_path, visited=seen),
        )
    return deep_merge(merged, data)


def _load_legacy_prompt_package_map(path: Path) -> Dict[str, Dict[str, Any]]:
    data = _read_yaml_mapping(path)
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"Prompt package '{key}' in {path} must be a mapping, got {type(value).__name__}"
            )
        out[str(key)] = value
    return out


def _iter_package_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    out: List[Path] = []
    for file_path in files:
        rel_parts = file_path.relative_to(root).parts
        if any(part.startswith("_") for part in rel_parts):
            continue
        out.append(file_path)
    return out


def load_prompt_package_map(prompt_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load all prompt packages from a directory or from a legacy single YAML file."""

    if prompt_root.is_file():
        return _load_legacy_prompt_package_map(prompt_root)

    if prompt_root.is_dir():
        packages: Dict[str, Dict[str, Any]] = {}
        for file_path in _iter_package_files(prompt_root):
            name = file_path.stem
            if name in packages:
                raise ValueError(f"Duplicate prompt package name '{name}' from file {file_path}")
            packages[name] = resolve_prompt_file(file_path)
        return packages

    raise FileNotFoundError(f"Prompt path does not exist: {prompt_root}")


def list_prompt_packages(prompt_root: Path) -> List[str]:
    """Return available prompt package names sorted alphabetically."""

    return sorted(load_prompt_package_map(prompt_root).keys())


def load_prompt_package(prompt_root: Path, package_name: str) -> Dict[str, Any]:
    """Load one prompt package by name and return it as a plain mapping."""

    packages = load_prompt_package_map(prompt_root)
    pkg = packages.get(str(package_name).strip())
    if not isinstance(pkg, dict):
        raise KeyError(f"Prompt package not found: {package_name}")
    return dict(pkg)


def save_prompt_package(prompt_root: Path, package_name: str, data: Dict[str, Any]) -> Path:
    """Persist a prompt package into a directory-based prompt store."""

    if prompt_root.is_file():
        raise ValueError(
            "Cannot save a single package into legacy prompts YAML. "
            "Set prompts.path to a directory first."
        )
    if not prompt_root.exists():
        prompt_root.mkdir(parents=True, exist_ok=True)

    name = str(package_name or "").strip()
    if not name:
        raise ValueError("Package name must not be empty")
    if "/" in name or "\\" in name:
        raise ValueError("Package name must not contain path separators")
    if name.startswith("_"):
        raise ValueError("Package name must not start with '_'")

    target = prompt_root / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return target
