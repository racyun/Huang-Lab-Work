#!/usr/bin/env python3
"""Pick the target and the input columns for an ExN model from exn_schema.json.

The label is chosen when the model runs, not when the tables are built: both
candidate labels live in exn_nodes.csv, and every model script takes

    --label {tagln_high, red}   which 0/1 column is the target
    --version {A, B}            A = morphology only
                                B = A + neighbours' marker readouts and labels
    --use-conditions            also give the model stiffness / treatment / ECM

and calls select_columns() to turn those flags into column lists. Column roles
come from exn_schema.json (written by extract_exn_features.py), so nothing is
picked by hand.

What each version may read
--------------------------
                      target cell          neighbour cells
    Version A         feature              feature
    Version B         feature              feature + marker + label
    (both)            + condition if --use-conditions (target cell only;
                        it is the same for every cell in a field)

A target cell's own marker and label columns are never inputs, in either
version, because both labels are defined from them. select_columns() checks
this and raises if it would be violated.

Check a plan without training anything:

    python exn_pipeline/exn_columns.py --schema <out>/exn_schema.json \\
        --label red --version B
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

# --label value -> 0/1 column in exn_nodes.csv
LABELS = {
    "tagln_high": "label_tagln_high",   # TAGLN_high == True            (~25% of cells)
    "red": "label_red",                 # TAGLN_high AND VE-cad-low     (~5% of cells)
}

# Columns that define a label. Checked independently of the schema's roles, so
# a mis-tagged schema still cannot put them among the target cell's inputs.
LABEL_SOURCES = {"TAGLN_high", "VEcad_bright_edge_fraction", "display_class"}

# Text / boolean columns with a 0/1 twin in the node table. Only the twin is
# used as a numeric input, so the same information does not enter twice.
TEXT_TWINS = {"TAGLN_high": "label_tagln_high", "display_class": "label_red"}


@dataclass
class ColumnPlan:
    label: str
    version: str
    target: str                                   # node column to predict
    self_inputs: list[str]                        # target cell's own columns
    neighbor_inputs: list[str]                    # read from each neighbour's node row
    edge_inputs: list[str]                        # read from the edge row
    excluded: dict[str, str] = field(default_factory=dict)   # column -> reason

    def summary(self) -> str:
        lines = [f"label={self.label}  version={self.version}  target column={self.target}",
                 f"  target-cell inputs ({len(self.self_inputs)}): {', '.join(self.self_inputs)}",
                 f"  neighbour inputs   ({len(self.neighbor_inputs)}): "
                 f"{', '.join(self.neighbor_inputs)}",
                 f"  edge inputs        ({len(self.edge_inputs)}): {', '.join(self.edge_inputs)}",
                 "  never inputs for the target cell:"]
        lines += [f"    {c:28s} {why}" for c, why in self.excluded.items()]
        return "\n".join(lines)


def add_model_args(ap: argparse.ArgumentParser) -> None:
    """Add --label / --version / --use-conditions to a model script's parser."""
    ap.add_argument("--label", required=True, choices=sorted(LABELS),
                    help="tagln_high = label_tagln_high (TAGLN_high), "
                         "red = label_red (display_class Red)")
    ap.add_argument("--version", required=True, choices=["A", "B"],
                    help="A = morphology only; B = A + neighbours' markers and labels")
    ap.add_argument("--use-conditions", action="store_true",
                    help="add stiffness / treatment / ECM as target-cell inputs")


def load_schema(path: Path | str) -> dict:
    schema = json.loads(Path(path).read_text())
    if "roles" not in schema:
        raise SystemExit(f"{path} has no column roles; rerun extract_exn_features.py")
    return schema


def select_columns(schema: dict, label: str, version: str,
                   use_conditions: bool = False) -> ColumnPlan:
    if label not in LABELS:
        raise ValueError(f"--label must be one of {sorted(LABELS)}, got {label!r}")
    if version not in ("A", "B"):
        raise ValueError(f"--version must be A or B, got {version!r}")

    roles = {c: v["role"] for c, v in schema["nodes"].items()}
    by_role = lambda r: [c for c, x in roles.items() if x == r]   # noqa: E731
    target = LABELS[label]
    if roles.get(target) != "label":
        raise ValueError(f"{target} is not tagged as a label in the schema")

    features, conditions = by_role("feature"), by_role("condition")
    self_inputs = features + (conditions if use_conditions else [])
    neighbor_inputs = list(features)
    if version == "B":
        neighbor_inputs += [c for c in by_role("marker") + by_role("label")
                            if c not in TEXT_TWINS]
    edge_inputs = [c for c, v in schema["edges"].items() if v["role"] == "feature"]

    excluded = {}
    for c, r in roles.items():
        if c in self_inputs:
            continue
        if c == target:
            excluded[c] = "the target"
        elif r in ("marker", "label"):
            excluded[c] = (f"{r} of the target cell" +
                           (" (neighbours' value used)" if c in neighbor_inputs else ""))
        elif r == "condition":
            excluded[c] = "condition (add with --use-conditions)"
        else:
            excluded[c] = r

    # leakage guard: nothing that defines either label may describe the target cell
    leaked = [c for c in self_inputs
              if roles[c] in ("marker", "label", "unreviewed")
              or c in LABEL_SOURCES or c.startswith("label_")]
    if leaked:
        raise AssertionError(f"target-cell inputs include marker/label columns: {leaked}")
    return ColumnPlan(label, version, target, self_inputs, neighbor_inputs, edge_inputs,
                      excluded)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the columns a model would use.")
    ap.add_argument("--schema", type=Path, required=True, help="exn_schema.json")
    add_model_args(ap)
    args = ap.parse_args()
    plan = select_columns(load_schema(args.schema), args.label, args.version,
                          args.use_conditions)
    print(plan.summary())


if __name__ == "__main__":
    main()
