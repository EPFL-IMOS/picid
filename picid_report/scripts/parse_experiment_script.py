"""
Parse experiment .sh scripts to extract datasets=(), model_names=(), command_to_run=(), and wandb_log_folder.

Used by the reproduction script generator to reuse the exact command shape from the scripts.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ParsedScript:
    """Result of parsing an experiment script."""

    path: Path
    datasets: List[str]
    model_names: List[str]
    command_lines: List[str]
    wandb_log_folder: Optional[str] = None
    seeds: Optional[List[int]] = None  # from SEEDS=( ... ), used for baselines


def _find_array_block(content: str, start_marker: str) -> Tuple[int, int]:
    """Find the span of an array: from 'start_marker' to the matching closing ')' (same line or next lines)."""
    idx = content.find(start_marker)
    if idx == -1:
        return -1, -1
    start = idx + len(start_marker)
    depth = 1
    i = start
    while i < len(content) and depth > 0:
        if content[i] == "(":
            depth += 1
        elif content[i] == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return idx, -1
    return idx, i - 1


def _parse_array_lines(content: str, start: int, end: int) -> List[str]:
    """Extract quoted or unquoted lines from an array block. Strips comments and empty lines."""
    block = content[start:end]
    lines: List[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip("\\").strip()
        if not line or line.startswith("#"):
            continue
        # Remove trailing comma if present
        if line.endswith(","):
            line = line[:-1].strip()
        # Unquote if wrapped in double quotes
        if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if line:
            lines.append(line)
    return lines


def _parse_command_block(content: str, start: int, end: int) -> List[str]:
    """Extract the command array lines (each line is one token, e.g. 'python picid/run.py \\' or 'experiment=${exp_name} \\')."""
    block = content[start:end]
    lines: List[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip("\\").strip()
        if not line:
            continue
        if line.startswith("#"):
            lines.append(raw_line.strip())
            continue
        if line.endswith(","):
            line = line[:-1].strip()
        if line:
            lines.append(line)
    return lines


def _parse_seeds_array(content: str) -> Optional[List[int]]:
    """Extract SEEDS=( ... ) as list of integers. Returns None if not found or empty."""
    s_start, s_end = _find_array_block(content, "SEEDS=(")
    if s_start == -1 or s_end == -1:
        return None
    block = content[s_start + len("SEEDS=(") : s_end]
    nums: List[int] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip("\\").strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        for part in line.split():
            try:
                nums.append(int(part))
            except ValueError:
                pass
    return nums if nums else None


def _extract_wandb_log_folder(content: str) -> Optional[str]:
    """Extract wandb_log_folder=\"...\" or wandb_log_folder=... from script."""
    # Match wandb_log_folder="29_01_2026" or wandb_log_folder=29_01_2026
    m = re.search(r'wandb_log_folder\s*=\s*["\']?([^"\'\s\n]+)["\']?', content)
    if m:
        return m.group(1).strip()
    return None


def parse_experiment_script(script_path: str | Path) -> Optional[ParsedScript]:
    """
    Parse an experiment .sh script.

    Extracts:
    - datasets=( ... )  -> list of dataset strings (e.g. "nb14|combined|prognostics")
    - model_names=( ... ) -> list of model names (e.g. lstm, tabpfn_fit_predict)
    - command_to_run=( ... ) -> list of command lines (exact content for substitution)
    - wandb_log_folder=... -> optional string

    Returns
    -------
    ParsedScript or None if required blocks are missing.
    """
    path = Path(script_path)
    if not path.exists():
        return None
    content = path.read_text()

    wandb_log_folder = _extract_wandb_log_folder(content)

    # datasets=(
    d_start, d_end = _find_array_block(content, "datasets=(")
    if d_start == -1 or d_end == -1:
        return None
    datasets = _parse_array_lines(content, d_start + len("datasets=("), d_end)

    # model_names=(
    m_start, m_end = _find_array_block(content, "model_names=(")
    if m_start == -1 or m_end == -1:
        return None
    model_names = _parse_array_lines(content, m_start + len("model_names=("), m_end)

    # command_to_run=(
    c_start, c_end = _find_array_block(content, "command_to_run=(")
    if c_start == -1 or c_end == -1:
        return None
    command_lines = _parse_command_block(content, c_start + len("command_to_run=("), c_end)

    seeds = _parse_seeds_array(content)

    return ParsedScript(
        path=path,
        datasets=datasets,
        model_names=model_names,
        command_lines=command_lines,
        wandb_log_folder=wandb_log_folder,
        seeds=seeds,
    )
