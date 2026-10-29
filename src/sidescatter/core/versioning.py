"""Workspace file versions, and the upgrade steps between them.

Every .ssw file says which format it uses ("version") and which Sidescatter
wrote it ("written_by"). FORMAT_VERSION goes up by one only when the saved
structure changes in a way an older Sidescatter couldn't read correctly; a new
optional field doesn't need it (older files just don't have the field).

When it does go up, add an upgrade step below: a function that takes a
version-n document and returns it as version n + 1. Opening an older file runs
every step from its version up, in memory, so nothing changes on disk until the
workspace is saved (and then the original is kept as a backup).

Then make a new example workspace (scripts/make_example_workspace.py), so the
tests keep opening every older format.
"""

FORMAT_VERSION = 1
FORMAT_NAME = "sidescatter-workspace"
RELEASES_URL = "https://github.com/danielmoyer011/sidescatter/releases/latest"

# version n -> function upgrading a version-n document to version n + 1, e.g.
#   def _v1_to_v2(doc):
#       for g in doc["gates"]: ...
#       return doc
#   UPGRADES = {1: _v1_to_v2}
UPGRADES = {}


class NewerWorkspaceError(ValueError):
    """The file is from a newer Sidescatter than this one."""

    def __init__(self, version, written_by=None):
        self.version, self.written_by = version, written_by
        from .. import __version__

        by = f"Sidescatter {written_by}" if written_by else "a newer Sidescatter"
        super().__init__(
            f"This workspace was saved by {by}, which is newer than this copy "
            f"(Sidescatter {__version__}). Update Sidescatter to open it.")


def upgrade(doc):
    """Bring a workspace document up to FORMAT_VERSION. Returns (doc, the version
    it was saved in). Raises NewerWorkspaceError for files from a newer Sidescatter."""
    if doc.get("format") != FORMAT_NAME:
        raise ValueError("not a Sidescatter workspace")
    version = int(doc.get("version", 1))
    if version > FORMAT_VERSION:
        raise NewerWorkspaceError(version, doc.get("written_by"))
    saved_as = version
    while version < FORMAT_VERSION:
        doc = UPGRADES[version](doc)
        version += 1
        doc["version"] = version
    return doc, saved_as


# ----- format fingerprint (see tests/test_versioning.py) ----------------------

# Dicts at these places are keyed by data (sample ids, channel names...), not
# by field names, so the fingerprint merges their values.
_MAPS = {"transforms", "groups", "unmixers", "qc", "peaks", "margin_counts"}


def shape(value, key=""):
    """The structure of a document (field names and value types), ignoring data.
    Two documents with the same fields have the same shape."""
    if isinstance(value, dict):
        if key in _MAPS:
            merged = {}
            for v in value.values():
                merged = _merge(merged, shape(v))
            return {"{map}": merged}
        return {k: shape(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        merged = None
        for v in value:
            s = shape(v, key)
            merged = s if merged is None else _merge(merged, s)
        return [merged] if merged is not None else []
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _merge(a, b):
    if a == b or not a:
        return b
    if not b:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a.get(k), b.get(k)) if k in a and k in b else a.get(k, b.get(k))
                for k in sorted(set(a) | set(b))}
    if isinstance(a, list) and isinstance(b, list):
        return [_merge(a[0] if a else None, b[0] if b else None)]
    if "null" in (a, b):
        return b if a == "null" else a
    if isinstance(a, str) and isinstance(b, str):
        return "|".join(sorted(set(a.split("|")) | set(b.split("|"))))
    return a
