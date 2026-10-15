#!/usr/bin/env python3
"""Expert triplets, step 2 — the annotation web app.

Shows three neighbourhood crops at a time. The expert clicks the one that does
not belong (or presses 1 / 2 / 3), or marks the triplet Unusable (X). Z undoes
the last answer. Standard library only (http.server + sqlite3), so it runs
anywhere the crops are, e.g. a Lightning studio with the port shared.

Where the answers go: <data-dir>/annotations.sqlite, three append-only tables.

  triplets      one row per triplet the app has ever created: its three crop
                ids, split (train / test), and whether it is shared (shown to
                every annotator, for inter-rater agreement).
  annotations   one row per answer: who, which triplet, the left-to-right
                order it was shown in, the position clicked and the crop id it
                maps to (or 'unusable'), response time, timestamps, and the
                app / instructions / render versions.
  retractions   undo marks an earlier answer as retracted instead of deleting
                it, so the log is never edited.

Motif labels are never stored here: the log only says which neighbourhoods an
expert thinks belong together, so it stays blind and still scores any future
model run. score_triplets.py exports it to CSV and computes the scores.

How triplets are chosen (all configurable):
  * an annotator's first --test-first answered triplets are test triplets;
    after that each new triplet is test with probability --test-share;
  * test triplets use three never-used test crops from three different images,
    so test triplets are independent of each other;
  * train triplets reuse train crops (least-used first), again three images;
  * about 1 in --shared-every test triplets is shared with all annotators.

Usage:
    python annotate_app.py --data-dir <root>/expert_triplets \
        --annotators "Prof. Huang,Carver" --host 0.0.0.0 --port 8765 \
        --token <a long random string>
Then share http://<host>:8765/?token=<token>.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

APP_VERSION = "triplets-app-v1"
INSTRUCTIONS_VERSION = "v1"
HERE = Path(__file__).resolve().parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS triplets (
    triplet_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    created_for  TEXT NOT NULL,
    split        TEXT NOT NULL CHECK (split IN ('train', 'test')),
    shared       INTEGER NOT NULL DEFAULT 0,
    crop_1       TEXT NOT NULL,
    crop_2       TEXT NOT NULL,
    crop_3       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    triplet_id            TEXT NOT NULL REFERENCES triplets(triplet_id),
    annotator             TEXT NOT NULL,
    shown_left            TEXT NOT NULL,
    shown_middle          TEXT NOT NULL,
    shown_right           TEXT NOT NULL,
    choice                TEXT NOT NULL
                          CHECK (choice IN ('left', 'middle', 'right', 'unusable')),
    odd_crop              TEXT,
    response_ms           INTEGER,
    shown_at              TEXT,
    answered_at           TEXT NOT NULL,
    app_version           TEXT,
    instructions_version  TEXT,
    render_version        TEXT
);
CREATE TABLE IF NOT EXISTS retractions (
    annotation_id  INTEGER PRIMARY KEY REFERENCES annotations(annotation_id),
    retracted_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ann_by_annotator ON annotations(annotator, triplet_id);
"""

LIVE = ("SELECT a.* FROM annotations a "
        "LEFT JOIN retractions r ON r.annotation_id = a.annotation_id "
        "WHERE r.annotation_id IS NULL")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    """All database access, behind one lock: two annotators must never be
    handed the same never-used test crop."""

    def __init__(self, data_dir: Path, args):
        self.dir = data_dir
        self.args = args
        self.lock = threading.Lock()
        self.db = sqlite3.connect(data_dir / "annotations.sqlite",
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.rng = random.Random(args.seed)
        self._crops_mtime = None
        self.render_cfg = json.loads((data_dir / "render_config.json").read_text())
        self.render_version = self.render_cfg["render_version"]
        self._load_crops()

    # ---- crop buffer (re-read when prepare_crops.py appends a batch) ----
    def _load_crops(self):
        path = self.dir / "crops.csv"
        mtime = path.stat().st_mtime
        if mtime == self._crops_mtime:
            return
        df = pd.read_csv(path, dtype={"image_id": str})
        self.crops = df.set_index("crop_id")
        self._crops_mtime = mtime

    def _usage(self) -> dict:
        used: dict = {}
        for r in self.db.execute("SELECT crop_1, crop_2, crop_3 FROM triplets"):
            for c in r:
                used[c] = used.get(c, 0) + 1
        return used

    def _pick(self, split: str) -> list | None:
        """Three crops from three different images of `split`."""
        df = self.crops[self.crops.split == split]
        used = self._usage()
        if split == "test":
            df = df[~df.index.isin(list(used))]          # each test crop once
            cands = list(df.index)
            self.rng.shuffle(cands)
        else:
            cands = list(df.index)
            self.rng.shuffle(cands)
            cands.sort(key=lambda c: used.get(c, 0))     # least-used first
        chosen, images = [], set()
        for c in cands:
            img = (df.at[c, "condition"], df.at[c, "image_id"])
            if img in images:
                continue
            chosen.append(c)
            images.add(img)
            if len(chosen) == 3:
                return chosen
        return None

    def _count_test(self, who: str) -> int:
        q = (f"SELECT COUNT(*) FROM ({LIVE}) a JOIN triplets t "
             f"ON t.triplet_id = a.triplet_id WHERE a.annotator = ? "
             f"AND t.split = 'test'")
        return self.db.execute(q, (who,)).fetchone()[0]

    def _payload(self, t: sqlite3.Row, order: list | None = None) -> dict:
        crops = [t["crop_1"], t["crop_2"], t["crop_3"]]
        if order is None:
            order = crops[:]
            self.rng.shuffle(order)                      # randomise position
        return {"triplet_id": t["triplet_id"], "order": order,
                "images": [f"/{self.crops.at[c, 'file']}" for c in order]}

    # ---- the three operations the page uses ----
    def next_for(self, who: str) -> dict:
        with self.lock:
            self._load_crops()
            # 1. anything already handed to this annotator but unanswered
            #    (page refresh, or an answer they undid)
            pending = self.db.execute(
                f"SELECT t.* FROM triplets t WHERE "
                f"(t.created_for = ? OR (t.shared = 1 AND t.created_for != ?)) "
                f"AND t.triplet_id NOT IN "
                f"(SELECT triplet_id FROM ({LIVE}) WHERE annotator = ?) "
                f"ORDER BY (t.created_for = ?) DESC, t.created_at LIMIT 1",
                (who, who, who, who)).fetchone()
            if pending is not None:
                return self._payload(pending)

            # 2. a new triplet
            a = self.args
            if self._count_test(who) < a.test_first:
                split = "test"
            else:
                split = "test" if self.rng.random() < a.test_share else "train"
            chosen = self._pick(split)
            if chosen is None and split == "test":
                print(f"[buffer] test crops exhausted; serving train to {who}. "
                      f"Run prepare_crops.py --n-test N to top up.",
                      file=sys.stderr)
                split, chosen = "train", self._pick("train")
            if chosen is None:
                return {"empty": True}
            shared = int(split == "test" and len(a.annotators) > 1
                         and self.rng.random() < 1.0 / a.shared_every)
            tid = uuid.uuid4().hex[:12]
            self.db.execute(
                "INSERT INTO triplets VALUES (?,?,?,?,?,?,?,?)",
                (tid, now(), who, split, shared, *chosen))
            self.db.commit()
            t = self.db.execute("SELECT * FROM triplets WHERE triplet_id = ?",
                                (tid,)).fetchone()
            return self._payload(t)

    def answer(self, body: dict) -> dict:
        who = str(body.get("annotator", "")).strip()
        tid = body.get("triplet_id")
        order = body.get("order") or []
        choice = body.get("choice")
        with self.lock:
            t = self.db.execute("SELECT * FROM triplets WHERE triplet_id = ?",
                                (tid,)).fetchone()
            if t is None or not who:
                return {"error": "unknown triplet or annotator"}
            if sorted(order) != sorted([t["crop_1"], t["crop_2"], t["crop_3"]]):
                return {"error": "order does not match the triplet"}
            if choice not in ("left", "middle", "right", "unusable"):
                return {"error": "bad choice"}
            if t["created_for"] != who and not t["shared"]:
                return {"error": "triplet belongs to another annotator"}
            dup = self.db.execute(
                f"SELECT 1 FROM ({LIVE}) WHERE annotator = ? AND triplet_id = ?",
                (who, tid)).fetchone()
            if dup:
                return {"ok": True, "duplicate": True}   # double-click
            odd = (None if choice == "unusable"
                   else order[["left", "middle", "right"].index(choice)])
            cur = self.db.execute(
                "INSERT INTO annotations (triplet_id, annotator, shown_left, "
                "shown_middle, shown_right, choice, odd_crop, response_ms, "
                "shown_at, answered_at, app_version, instructions_version, "
                "render_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, who, *order, choice, odd,
                 int(body.get("response_ms") or 0) or None,
                 body.get("shown_at"), now(), APP_VERSION,
                 INSTRUCTIONS_VERSION, self.render_version))
            self.db.commit()
            return {"ok": True, "annotation_id": cur.lastrowid}

    def undo(self, who: str) -> dict:
        with self.lock:
            last = self.db.execute(
                f"SELECT * FROM ({LIVE}) WHERE annotator = ? "
                f"ORDER BY annotation_id DESC LIMIT 1", (who,)).fetchone()
            if last is None:
                return {"error": "nothing to undo"}
            self.db.execute("INSERT INTO retractions VALUES (?, ?)",
                            (last["annotation_id"], now()))
            self.db.commit()
            t = self.db.execute("SELECT * FROM triplets WHERE triplet_id = ?",
                                (last["triplet_id"],)).fetchone()
            order = [last["shown_left"], last["shown_middle"], last["shown_right"]]
            return self._payload(t, order)

    def progress(self, who: str) -> dict:
        n = self.db.execute(f"SELECT COUNT(*) FROM ({LIVE}) WHERE annotator = ?",
                            (who,)).fetchone()[0]
        today = datetime.now(timezone.utc).date().isoformat()
        n_today = self.db.execute(
            f"SELECT COUNT(*) FROM ({LIVE}) WHERE annotator = ? "
            f"AND answered_at >= ?", (who, today)).fetchone()[0]
        return {"total": n, "today": n_today}


def make_handler(store: Store, token: str | None, page: bytes):
    crops_root = (store.dir / "crops").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):          # keep the console quiet
            pass

        def _authorised(self, q) -> bool:
            if not token:
                return True
            given = self.headers.get("X-Token") or (q.get("token") or [""])[0]
            return given == token

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(
                ("application/json", "text/html")) else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if not self._authorised(q):
                return self._json({"error": "missing or wrong token"}, 403)
            if u.path == "/api/config":
                return self._json({"annotators": store.args.annotators,
                                   "instructions_version": INSTRUCTIONS_VERSION})
            who = (q.get("annotator") or [""])[0].strip()
            if u.path == "/api/next" and who:
                return self._json(store.next_for(who))
            if u.path == "/api/progress" and who:
                return self._json(store.progress(who))
            if u.path.startswith("/crops/"):
                p = (store.dir / u.path.lstrip("/")).resolve()
                if crops_root in p.parents and p.suffix == ".png" and p.exists():
                    return self._send(200, p.read_bytes(), "image/png")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            if not self._authorised(parse_qs(u.query)):
                return self._json({"error": "missing or wrong token"}, 403)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if u.path == "/api/answer":
                return self._json(store.answer(body))
            if u.path == "/api/undo":
                return self._json(store.undo(str(body.get("annotator", "")).strip()))
            return self._json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    sys.path.insert(0, str(HERE.parents[0] / "stage4_motifs"))
    from plot_motif_maps import _default_local_root
    ap = argparse.ArgumentParser(description="Odd-one-out annotation app.")
    ap.add_argument("--data-dir", type=Path,
                    default=_default_local_root() / "expert_triplets")
    ap.add_argument("--annotators", default="Prof. Huang,Carver",
                    help="comma-separated names offered on the start screen")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None,
                    help="shared secret required in the URL (?token=...); "
                         "strongly recommended whenever --host is not local")
    ap.add_argument("--test-first", type=int, default=150,
                    help="answered test triplets per annotator before mixing")
    ap.add_argument("--test-share", type=float, default=0.5,
                    help="after --test-first, share of new triplets that are test")
    ap.add_argument("--shared-every", type=int, default=5,
                    help="about 1 in N test triplets goes to every annotator")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    args.annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]

    for need in ("crops.csv", "render_config.json"):
        if not (args.data_dir / need).exists():
            raise SystemExit(f"{args.data_dir / need} missing: run prepare_crops.py first")
    if args.host not in ("127.0.0.1", "localhost") and not args.token:
        print("[warning] serving beyond this machine without --token: anyone "
              "who can reach the port can annotate", file=sys.stderr)

    store = Store(args.data_dir, args)
    page = (HERE / "annotate.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(store, args.token, page))
    shown = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"Annotation app on http://{shown}:{args.port}/"
          f"{'?token=' + args.token if args.token else ''}")
    print(f"  logging to {args.data_dir / 'annotations.sqlite'}")
    print(f"  crops cut from {store.render_cfg.get('source_imgs_dir', '?')}")
    print(f"  cells from     {store.render_cfg.get('source_master_table', '?')}")
    print(f"  {len(store.crops)} crops in the buffer "
          f"({store.crops.split.value_counts().to_dict()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
