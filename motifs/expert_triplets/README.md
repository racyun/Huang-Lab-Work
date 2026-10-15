# Expert triplets: odd-one-out validation of the motifs

Prof. Huang and Carver see three real neighbourhood crops at a time and click
the one that doesn't belong, or mark the set **Unusable**. Their answers are
logged without motif labels, so the same log scores any model run: v4, a new k,
another edge rule, or the cell-only baseline.

Everything runs where the real data is (the Lightning studio). Crops are cut
from the original micrographs listed in the master table. There is no
synthetic or placeholder imagery anywhere in this pipeline.

## Run it

```bash
# 1. Freeze the train/test image split and render a crop buffer (real data).
python motifs/expert_triplets/prepare_crops.py --n-test 600 --n-train 1200

# 2. Serve the annotation page. Share the printed link (it includes the token).
python motifs/expert_triplets/annotate_app.py --host 0.0.0.0 --port 8765 \
    --annotators "Prof. Huang,Carver" --token "$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"

# 3. Whenever you want numbers (export + scores, test split by default).
python motifs/expert_triplets/score_triplets.py --data-dir <root>/expert_triplets \
    --motifs <root>/stage4/motifs.parquet --embeddings <root>/stage3/embeddings.parquet
```

Running out of crops? Run step 1 again with `--n-test` / `--n-train` to append
another batch. The app picks it up without a restart. There's no cap on how many
triplets get answered.

## What is stored, and where

All of it lives in `<root>/expert_triplets/` (default
`/teamspace/studios/this_studio/cellpose_work/expert_triplets/`). Back the
folder up to Drive with the other run outputs.

| File | Written by | Contents |
|---|---|---|
| `split.csv` | step 1, once | Each image assigned to train or test, stratified by condition. Never rewritten. |
| `render_config.json` | step 1, once | Crop size, contrast, marker, and the source master table and image folder. Frozen so every crop has the same scale. |
| `crops.csv` + `crops/` | step 1, appended | One row and one PNG per crop: image, cell id, centroid, window, batch, seed, file checksum. |
| `annotations.sqlite` | the app, appended | `triplets` (every triplet created), `annotations` (every answer), `retractions` (undos). Rows are never edited or deleted. |
| `annotations_export.csv` | step 3 | Flat CSV of all live answers joined to crop metadata. The portable record. |
| `triplet_scores.json` | step 3 | The latest scores. |

Each answer records: the annotator, the triplet, the left/middle/right order it
was shown in, the position clicked and the crop it maps to (or `unusable`),
response time, timestamps, and app/instructions/render versions.

## How triplets are drawn

- **Split by image.** Test and train crops never come from the same image, and
  a triplet never mixes the two.
- **Test first.** Each annotator's first 150 answered triplets are test; after
  that, new triplets are test half the time (`--test-first`, `--test-share`).
- **Test triplets are independent.** Each uses three never-used test crops from
  three different images. Train triplets reuse train crops, least-used first.
- **Shared triplets.** About 1 in 5 test triplets goes to both annotators
  (`--shared-every`). Their agreement is the human ceiling.
- **Position is randomised** every time a triplet is shown.

## How the answers are used

**Evaluation (test split).** `score_triplets.py` reports:

- **Motif agreement.** Credit 1 if the expert's pair shares a motif and the odd
  crop doesn't, 0 if the motifs contradict the expert, 1/3 if the motifs have no
  opinion (all same or all different). Chance is exactly 1/3, so neither very
  few nor very many motifs can game it. A permutation null is reported too.
- **Human ceiling**, and the **normalised score**
  (score − 1/3) / (ceiling − 1/3), shown once there are 30+ shared triplets.
- **Embedding accuracy.** Is the odd crop the one left out of the closest pair
  in the GNN embedding?
- **Odd-one-out loss.** Negative log-likelihood of the experts' answers under
  P(odd = k) ∝ exp(−d²(other two) / τ), with τ fitted on train answers.
  Uniform guessing gives 1.0986. Lower is better.

**Intervals.** The *anytime* 95% interval stays valid however often you look
while annotation continues. The bootstrap interval is only valid if you analyse
once, at a sample size fixed in advance.

**Training (train split).** `triplet_metrics.torch_odd_one_out_loss` is the same
loss in differentiable form, for fitting per-dimension weights on the frozen
embeddings. Pass the fitted weights back to the scorer with `--weights` and
compare test scores before and after. Never fit on test answers.
