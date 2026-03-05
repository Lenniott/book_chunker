#!/usr/bin/env python3
"""
convert_md.py

Usage:
    python convert_md.py book.json 2.3.1 32.1

Loads a structured book JSON file, finds the section specified by the
node ID/path (e.g., "2.4.1"), and writes a Markdown file with standard
metadata front matter plus the section's content rendered in Markdown form.
When no explicit output path is provided, the script looks for a folder
within the configured Obsidian vault path whose name starts with the
metadata ID (e.g., "32.1") and writes the Markdown inside it.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonNode = Dict[str, Any]

DEFAULT_OBSIDIAN_BOOKS_DIR = Path(
    "/Users/benjamin/Library/Mobile Documents/iCloud~md~obsidian/Documents/"
    "Gilgamesh_house/30-39 Resources/32 Books"
)


def load_outline(json_path: Path) -> List[JsonNode]:
    """Read the JSON file and return its outline list."""
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as err:
        raise SystemExit(f"JSON file not found: {json_path}") from err
    except json.JSONDecodeError as err:
        raise SystemExit(f"Invalid JSON ({json_path}): {err}") from err

    if "outline" not in data:
        raise SystemExit(f"`outline` key not found in {json_path}")

    outline = data["outline"]
    if not isinstance(outline, list):
        raise SystemExit("`outline` must be a list")

    return outline


def iter_nodes(
    nodes: Iterable[JsonNode], parent_path: Optional[str] = None
) -> Iterable[Tuple[str, JsonNode]]:
    """
    Yield (node_path, node) pairs for every node in the outline tree.

    Node paths use dotted numbering (e.g., "1.2.3").
    """
    for index, node in enumerate(nodes, start=1):
        node_path = f"{parent_path}.{index}" if parent_path else str(index)
        node_id = node.get("node_id") or node_path
        yield node_id, node

        # Gather children defined either within `content` (dict entries)
        # or within the `sections` list in hierarchy-aware outputs.
        child_nodes: List[JsonNode] = []

        content_items = node.get("content") if isinstance(node, dict) else None
        if isinstance(content_items, list):
            child_nodes.extend(item for item in content_items if isinstance(item, dict))

        section_items = node.get("sections") if isinstance(node, dict) else None
        if isinstance(section_items, list):
            child_nodes.extend(item for item in section_items if isinstance(item, dict))

        if child_nodes:
            yield from iter_nodes(child_nodes, node_path)


def find_node_by_id(outline: List[JsonNode], node_id: str) -> JsonNode:
    """Locate a node by its dotted path ID."""
    for path, node in iter_nodes(outline):
        if path == node_id:
            return node
    raise SystemExit(f"ID `{node_id}` not found in outline")


def render_node_to_markdown(node: JsonNode, heading_level: int = 1) -> str:
    """
    Convert a node (and its children) to Markdown.

    Strings found in `content` become paragraphs. Child dict items and
    entries from `sections` recurse, producing deeper headings.
    """
    lines: List[str] = []
    title = node.get("title")

    if title:
        level = min(heading_level, 6)
        lines.append(f"{'#' * level} {title.strip()}")
        lines.append("")

    content_items = node.get("content")
    if isinstance(content_items, list):
        for item in content_items:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    lines.append(cleaned)
                    lines.append("")
            elif isinstance(item, dict):
                lines.append(render_node_to_markdown(item, heading_level + 1))
                lines.append("")

    section_items = node.get("sections")
    if isinstance(section_items, list):
        for child in section_items:
            if isinstance(child, dict):
                lines.append(render_node_to_markdown(child, heading_level + 1))
                lines.append("")

    # Remove trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def build_front_matter(meta_id: str, timestamp: datetime) -> str:
    """Return the metadata front matter block."""
    created = timestamp.strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        "area: 30-39 Resources",
        "category: 32 Books",
        f"id: {meta_id}",
        "type:",
        "status: draft",
        "linked: []",
        "tags: []",
        f"created: {created}",
        "archive_by:",
        "---",
        "",
    ]
    return "\n".join(lines)


def sanitize_filename(text: str) -> str:
    """Create a safe filename fragment from the node title."""
    if not text:
        return "section"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text.strip())
    return "-".join(filter(None, safe.split("-"))).lower() or "section"


def find_target_folder(base_dir: Path, prefix: str) -> Optional[Path]:
    """Return the first directory in base_dir whose name starts with prefix."""
    if not base_dir.exists():
        raise SystemExit(f"Books directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise SystemExit(f"Books directory is not a folder: {base_dir}")

    directories = sorted(
        p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    return directories[0] if directories else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert book JSON sections to Markdown.")
    parser.add_argument("json_path", help="Path to the book_structure JSON file")
    parser.add_argument("node_id", help='Section ID/path (e.g., "2.3.1")')
    parser.add_argument(
        "metadata_id",
        help='Metadata/filing ID (e.g., "32.1") used in front matter and folder lookup',
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output Markdown path. Defaults to <node_id>.md in current directory.",
    )
    parser.add_argument(
        "--books-dir",
        help="Override base directory for book folders",
        default=str(DEFAULT_OBSIDIAN_BOOKS_DIR),
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    outline = load_outline(json_path)
    node = find_node_by_id(outline, args.node_id)

    timestamp = datetime.now()
    actual_node_id = node.get("node_id") or args.node_id
    meta_id = args.metadata_id
    front_matter = build_front_matter(meta_id, timestamp)
    body = render_node_to_markdown(node).strip()
    output_text = f"{front_matter}{body}\n"

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        books_dir = Path(args.books_dir).expanduser().resolve()
        target_folder = find_target_folder(books_dir, meta_id)
        if target_folder is None:
            raise SystemExit(
                f"No folder found in {books_dir} starting with '{meta_id}'. "
                "Specify --output or create the folder."
            )
        title_fragment = sanitize_filename(node.get("title", ""))
        output_name = f"{meta_id}_{title_fragment}.md"
        output_path = target_folder / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(output_text)

    print(f"Wrote Markdown to {output_path}")


if __name__ == "__main__":
    main()

