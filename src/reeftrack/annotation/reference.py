"""Other annotators' labels on the same video, read-only, as a reference while you work.

Library runs keep one file per annotator (`behaviours_<name>.csv`, `loops_<name>.csv`). The GUI
edits only your own; this module reads everyone else's with the plain csv module, so their files
are never opened through a writable log (which can re-save on load) and are never changed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from reeftrack.io.run import slugify

KINDS = ("behaviours", "loops")
OUT_OF_VIEW = "Out of view"


@dataclass(frozen=True)
class RefEvent:
    kind: str  # "behaviours" | "loops"
    who: str  # annotator as written in the file (falls back to the file-name part)
    start: int
    end: int
    label: str  # the behaviour, or "Loop <id>"

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end


def other_annotators(run, me: str | None) -> dict[str, str]:
    """{file slug: display name} of everyone else with behaviour/loop files on this video."""
    if not run.is_library:
        return {}
    mine = slugify(me) if me else None
    out: dict[str, str] = {}
    for kind in KINDS:
        for p in sorted(run.root.glob(f"{kind}_*.csv")):
            slug = p.stem[len(kind) + 1:]
            if slug != mine and slug not in out:
                out[slug] = _display_name(p) or slug
    return dict(sorted(out.items(), key=lambda kv: kv[1].lower()))


def _display_name(path: Path) -> str:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("annotator") or "").strip():
                    return r["annotator"].strip()
    except (OSError, csv.Error, UnicodeDecodeError):
        pass
    return ""


def load_reference(run, slugs: list[str], n_frames: int) -> list[RefEvent]:
    """Every behaviour / loop event of the given annotators (by file slug). Unreadable or
    malformed rows are skipped rather than failing: this is only a reference."""
    out: list[RefEvent] = []
    for slug in slugs:
        for kind in KINDS:
            p = run.root / f"{kind}_{slug}.csv"
            if not p.exists():
                continue
            try:
                with open(p, newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except (OSError, csv.Error, UnicodeDecodeError):
                continue
            for r in rows:
                try:
                    s, e = int(r["start_frame"]), int(r["end_frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                s, e = max(0, min(s, e)), min(n_frames - 1, max(s, e))
                if s > e:
                    continue
                label = r.get("behaviour", "") if kind == "behaviours" else f"Loop {r.get('loop_id') or r.get('event_id', '')}"
                who = (r.get("annotator") or "").strip() or slug
                out.append(RefEvent(kind, who, s, e, label))
    return sorted(out, key=lambda ev: (ev.start, ev.end, ev.who))


def next_out_of_view(events: list[RefEvent], frame: int, direction: int = +1) -> int | None:
    """Start frame of the next (direction +1) / previous (-1) 'Out of view' stretch in `events`."""
    starts = sorted({ev.start for ev in events if ev.kind == "behaviours" and ev.label == OUT_OF_VIEW})
    if direction > 0:
        return next((s for s in starts if s > frame), None)
    return next((s for s in reversed(starts) if s < frame), None)
