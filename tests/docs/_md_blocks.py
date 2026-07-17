"""Shared helpers for the documentation reproduce-example test.

The reproduce test treats the repository ``README.md`` as executable truth: it
locates the fenced ``bash`` code blocks so the test module can execute the
published "Quick reproduce" recipe in its offline-safe form and fail CI if the
recipe drifts from the code.

This mirrors the *shape* of engrava's ``tests/docs/_md_blocks.py`` (fence
extraction driven by an anchor that lives in the test module — never a marker in
the published Markdown) but is deliberately thin: this repo only needs shell-block
extraction, not the Python compile/execute machinery.

A fenced block is recognised by an opening line whose first non-space token is
```` ```bash ```` (extra info-string words are ignored, as renderers do) and a
closing line that is exactly ```` ``` ````. Indented blocks are supported; the
captured body is dedented to the fence's indentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ``tests/docs/_md_blocks.py`` -> the repository root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

_BASH_FENCE = "```bash"
_CLOSING_FENCE = "```"


class DocBlockError(ValueError):
    """Raised when the README does not contain exactly one matching code block."""


@dataclass(frozen=True)
class ShellBlock:
    """A fenced ``bash`` code block extracted from a Markdown file.

    Attributes:
        path: Absolute path to the source Markdown file.
        rel: Path relative to the repository root (for messages).
        start_line: 1-based line number of the first body line.
        body: The dedented block body (without the fences).

    """

    path: Path
    rel: str
    start_line: int
    body: str

    @property
    def location(self) -> str:
        """Return a human-readable ``file:line`` locator."""
        return f"{self.rel}:{self.start_line}"

    @property
    def command_lines(self) -> list[str]:
        """Return the block's non-blank, non-comment command lines, stripped.

        Each line has any trailing ``# ...`` inline comment removed and is
        whitespace-stripped; blank lines and whole-line comments are dropped. None
        of the documented commands contain a literal ``#``, so splitting on it is
        safe and yields the bare shell command a reader would run.

        Returns:
            The recipe's command lines in document order.

        """
        commands: list[str] = []
        for raw in self.body.splitlines():
            code = raw.split("#", 1)[0].strip()
            if code:
                commands.append(code)
        return commands


def extract_shell_blocks(path: Path) -> list[ShellBlock]:
    """Extract every fenced ``bash`` code block from one Markdown file.

    Args:
        path: The Markdown file to scan.

    Returns:
        A list of :class:`ShellBlock` in document order. Empty when the file
        contains no ``bash`` fences.

    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()

    blocks: list[ShellBlock] = []
    in_block = False
    indent = 0
    body_lines: list[str] = []
    body_start = 0

    for index, raw in enumerate(lines):
        stripped = raw.lstrip()
        if not in_block:
            if stripped.startswith(_BASH_FENCE):
                in_block = True
                indent = len(raw) - len(stripped)
                body_lines = []
                body_start = index + 2  # 1-based, first line after the fence
            continue
        if stripped == _CLOSING_FENCE:
            body = "\n".join(_dedent(line, indent) for line in body_lines)
            blocks.append(ShellBlock(path=path, rel=rel, start_line=body_start, body=body))
            in_block = False
            continue
        body_lines.append(raw)

    return blocks


def block_with_anchor(path: Path, anchor: str) -> ShellBlock:
    """Return the single ``bash`` block in ``path`` whose body contains ``anchor``.

    The anchor lives in the test module (never in the published Markdown), so the
    binding is robust to line-number drift and documents *which* block is meant.

    Args:
        path: The Markdown file to scan.
        anchor: A substring that must appear in exactly one ``bash`` block.

    Returns:
        The uniquely matching block.

    Raises:
        DocBlockError: If zero or more than one block contains ``anchor``.

    """
    matches = [b for b in extract_shell_blocks(path) if anchor in b.body]
    if len(matches) != 1:
        rel = path.relative_to(REPO_ROOT).as_posix()
        msg = (
            f"expected exactly one fenced bash block in {rel} containing anchor "
            f"{anchor!r}, found {len(matches)}; update the anchor in the docs test "
            f"or restore the reproduce recipe"
        )
        raise DocBlockError(msg)
    return matches[0]


def _dedent(line: str, indent: int) -> str:
    """Strip up to ``indent`` leading spaces from a captured body line."""
    prefix = line[:indent]
    if prefix.strip() == "":
        return line[indent:]
    return line.lstrip()
