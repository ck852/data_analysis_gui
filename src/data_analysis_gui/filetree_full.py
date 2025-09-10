#!/usr/bin/env python3
# filetree.py
import os, sys, argparse, fnmatch, ctypes

# Defaults: exclude caches, VCS metadata, temp clutter; DO NOT exclude csv/mat/txt
DEFAULT_EXCLUDES = [
    # Python caches
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".coverage", "htmlcov",
    # VCS / IDE / tooling
    ".git", ".hg", ".svn", ".idea", ".vscode", "*.egg-info",
    # Envs / build artifacts
    ".venv", "venv", "env", ".env", ".tox", "build", "dist",
    # OS junk
    ".DS_Store", "Thumbs.db", "desktop.ini",
    # Other big vendor dirs (optional but common)
    "node_modules",
]
DEFAULT_OUTPUT = "filetree.txt"

def is_hidden(path: str) -> bool:
    """Dotfiles considered hidden; also honor Windows hidden/system attrs."""
    name = os.path.basename(path.rstrip(os.sep))
    if name.startswith('.'):
        return True
    if os.name == 'nt':
        try:
            FILE_ATTRIBUTE_HIDDEN = 0x2
            FILE_ATTRIBUTE_SYSTEM = 0x4
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1:
                return False
            return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))
        except Exception:
            return False
    return False

def _casefold(s: str) -> str:
    return s.casefold() if hasattr(str, "casefold") else s.lower()

def should_exclude(rel_path: str, name: str, patterns) -> bool:
    """Match either the basename or relative path (case-insensitive, glob)."""
    rp = rel_path.replace("\\", "/")
    rp_cf = _casefold(rp)
    base_cf = _casefold(name)
    for pat in patterns:
        pat_cf = _casefold(pat)
        if fnmatch.fnmatch(rp_cf, pat_cf) or fnmatch.fnmatch(base_cf, pat_cf):
            return True
    # Strong rule for __pycache__ directories
    if name == "__pycache__":
        return True
    return False

def scandir_sorted(path):
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except PermissionError:
        return []
    # dirs first, case-insensitive
    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
    return entries

def draw_tree(start=".", max_depth=None, show_all=False, dirs_only=False,
              excludes=None, file=sys.stdout):
    excludes = excludes or []
    start = os.path.abspath(start)
    root_name = os.path.basename(start) or start

    # ASCII connectors (txt-safe)
    tee, last, pipe, space = "|-- ", "`-- ", "|   ", "    "

    def _walk(dir_path, prefix, depth, rel_base):
        if max_depth is not None and depth > max_depth:
            return
        entries = scandir_sorted(dir_path)

        filtered = []
        for e in entries:
            name = e.name
            rel = os.path.join(rel_base, name) if rel_base else name
            if not show_all and is_hidden(e.path):
                continue
            if should_exclude(rel, name, excludes):
                continue
            if dirs_only and not e.is_dir(follow_symlinks=False):
                continue
            filtered.append(e)

        for i, entry in enumerate(filtered):
            connector = last if i == len(filtered) - 1 else tee
            display = entry.name + ("/" if entry.is_dir(follow_symlinks=False) else "")
            try:
                print(prefix + connector + display, file=file)
            except BrokenPipeError:
                return
            if entry.is_dir(follow_symlinks=False):
                new_prefix = prefix + (space if i == len(filtered) - 1 else pipe)
                _walk(entry.path, new_prefix, depth + 1,
                      os.path.join(rel_base, entry.name) if rel_base else entry.name)

    print(f"{root_name}/", file=file)
    _walk(start, "", 1, "")

def main():
    p = argparse.ArgumentParser(
        description=("Generate an ASCII file tree (Windows/macOS/Linux). "
                     "Defaults exclude caches/VCS/IDE/OS junk but include *.csv, *.mat, *.txt. "
                     "Writes to filetree.txt in the current directory."))
    p.add_argument("path", nargs="?", default=".", help="Root path (default: .)")
    p.add_argument("-L", "--depth", type=int, default=None, help="Max depth (like tree -L)")
    p.add_argument("-a", "--all", action="store_true", help="Include hidden files")
    p.add_argument("-d", "--dirs-only", action="store_true", help="List directories only")
    p.add_argument("--no-default-excludes", action="store_true",
                   help="Do not apply built-in excludes")
    p.add_argument("--exclude", action="append", default=[],
                   help="Additional glob to exclude (repeatable)")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                   help=f"Output file name (default: {DEFAULT_OUTPUT})")
    args = p.parse_args()

    excludes = [] if args.no_default_excludes else list(DEFAULT_EXCLUDES)
    if args.exclude:
        excludes.extend(args.exclude)

    out_path = os.path.join(os.path.abspath(args.path), args.output)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        draw_tree(args.path, args.depth, args.all, args.dirs_only, excludes, f)

    print(f"Wrote file tree to {out_path}")

if __name__ == "__main__":
    main()
