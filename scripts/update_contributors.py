#!/usr/bin/env python3
"""Regenerate the Contributors table in README.md from git log --all --numstat."""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

GITHUB_HANDLES = {
    "youwillfindinfinity": "youwillfindinfinity",
}

# Git author names that should be merged into a single canonical display name.
NAME_ALIASES: dict[str, str] = {
    "Zoe Azra Blei": "Zoe Azra",
}

MEDALS = ["🥇", "🥈", "🥉"]


def get_contributions() -> dict[str, dict]:
    result = subprocess.run(
        ["git", "log", "--all", "--numstat", "--format=COMMIT:%aN"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    stats: dict[str, dict] = defaultdict(lambda: {"added": 0, "deleted": 0})
    current_author = None
    for line in result.stdout.splitlines():
        if line.startswith("COMMIT:"):
            raw = line[7:].strip()
            current_author = NAME_ALIASES.get(raw, raw)
        elif current_author and line.strip() and not line.startswith("-"):
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    stats[current_author]["added"] += int(parts[0])
                    stats[current_author]["deleted"] += int(parts[1])
                except ValueError:
                    pass  # binary files show "-"
    return stats


def build_table(stats: dict) -> str:
    rows = sorted(
        stats.items(),
        key=lambda kv: kv[1]["added"] + kv[1]["deleted"],
        reverse=True,
    )
    grand_total = sum(v["added"] + v["deleted"] for v in stats.values()) or 1

    lines = [
        "| Rank | Contributor | Total Lines | Added | Deleted | Share |",
        "|:----:|-------------|------------:|------:|--------:|------:|",
    ]
    for i, (name, v) in enumerate(rows):
        total = v["added"] + v["deleted"]
        share = total / grand_total * 100
        medal = MEDALS[i] if i < len(MEDALS) else str(i + 1)
        handle = GITHUB_HANDLES.get(name)
        display = f"[{name}](https://github.com/{handle})" if handle else name
        lines.append(
            f"| {medal} | {display} | {total:,} | +{v['added']:,} | −{v['deleted']:,} | {share:.1f}% |"
        )
    return "\n".join(lines)


def patch_readme(table: str) -> None:
    text = README.read_text()
    new_section = (
        "## Contributors\n\n"
        "All-time line contributions (`git log --all --numstat`, added + deleted):\n\n"
        + table
    )
    patched = re.sub(
        r"## Contributors\n.*?(?=\n---|\Z)",
        new_section,
        text,
        flags=re.DOTALL,
    )
    README.write_text(patched)


if __name__ == "__main__":
    stats = get_contributions()
    table = build_table(stats)
    patch_readme(table)
    print("README.md contributors table updated.")
