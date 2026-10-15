# Expert triplets: odd-one-out validation of the motifs

Prof. Huang and Carver see three real neighbourhood crops at a time and click
the one that doesn't belong, or mark the set **Unusable**. The site is a Google
Apps Script web app attached to a Google Sheet. It's always on, free, and needs
no server: Google hosts the page, the Sheet stores the answers, and the crop
images stay private in your Drive.

Crops are cut from the original micrographs listed in the master table. There
is no synthetic or placeholder imagery anywhere in this pipeline.

| File | Where it runs | Job |
|---|---|---|
| `prepare_crops.py` | Lightning, once (plus top-ups) | Freeze the train/test image split and cut the crops |
| `sheets_app/Code.gs`, `sheets_app/Index.html` | Google Apps Script | The annotation site and its storage |
| `score_triplets.py` | Lightning or laptop, whenever | Score a model run against the answers |
| `triplet_metrics.py` | (imported) | Scores, the odd-one-out loss, intervals |

## One-time setup (about 20 minutes, mostly waiting on step 1)

**1. Cut the crops on Lightning.** Try a small batch first and look at a few
images, then top up. Crop size, contrast and image format freeze on the first
run.

```bash
cd ~/Huang-Lab-Work && git pull
python motifs/expert_triplets/prepare_crops.py --n-test 30
# look at a few files in cellpose_work/expert_triplets/crops/test/, then:
python motifs/expert_triplets/prepare_crops.py --n-test 570 --n-train 1200
```

**2. Copy the folder to Google Drive**, keeping its layout (`crops.csv`,
`render_config.json`, `split.csv`, `crops/test/`, `crops/train/`). With the
rclone remote you already use for Drive:

```bash
rclone copy ~/cellpose_work/expert_triplets "<your-remote>:<lab folder>/expert_triplets"
```

**3. Create the Sheet and paste in the code.** Make a new Google Sheet, for
example "Huang Lab triplet annotations", then open **Extensions > Apps Script**:
- Replace everything in `Code.gs` with this repo's `sheets_app/Code.gs`.
- Click **+ > HTML**, name it `Index` (no extension), and paste in
  `sheets_app/Index.html`.
- Save, then reload the Sheet. A **Triplets** menu appears.

**4. Set up and import.** In the Sheet:
- **Triplets > 1. Set up.** Paste the Drive folder link from step 2 and the
  annotator names. Google asks you to authorise the script. Because it's your
  own unpublished script, it shows "Google hasn't verified this app": choose
  *Advanced > Go to … (unsafe)*. The final dialog shows the **access key**.
- **Triplets > 2. Import crops from Drive.** This takes a minute or two for
  about 1,800 images.

**5. Deploy the site.** In Apps Script, go to **Deploy > New deployment >
Web app**:
- *Execute as:* **Me**. The script reads your Drive, so they never need access to it.
- *Who has access:* **Anyone**, so no Google sign-in is needed. If the lab's Google
  Workspace hides that option, choose your organisation instead, and they sign
  in with their Workspace accounts.
- Copy the web app URL. **The link to share is `<URL>?token=<access key>`.**
  Without the key the page refuses to load. *Triplets > Show settings* shows
  the key again.

**6. Try it, then clear your trial answers.** Open the link and pick a name
like "Rachel (trial)". Before sending it out, delete every row *below the
header* in the `triplets`, `annotations` and `retractions` tabs, so your trial
answers don't use up test crops or end up in the data.

## While they annotate

- **Nothing to keep running.** The site works whenever they open it, for as many
  days as they like. Every answer is saved immediately.
- **More crops:** run step 1 again with `--n-test N --n-train N`, rerun the
  `rclone copy` (it only adds new files), then *Triplets > 2. Import crops*.
- **Changing the code later:** after editing in Apps Script, go to *Deploy >
  Manage deployments > edit > Version: New version*. Otherwise the site keeps
  serving the old code.
- **Never sort, edit or delete rows** in the tabs once real annotation starts.
  The tabs are the record.

## What is stored

| Tab | Contents |
|---|---|
| `crops` | `crops.csv` plus each image's Drive file id |
| `triplets` | Every set ever shown: its three crop ids, train/test, shared or not |
| `annotations` | Every answer: who, which set, left/middle/right order shown, the position clicked and the crop it maps to (or `unusable`), response time, timestamps, versions |
| `retractions` | Undos. An answer is withdrawn, not deleted. |

Motif labels are never stored, so the same answers score any model run: v4,
a new k, other edge rules, or the cell-only baseline. Values are written as
plain text so ids and timestamps are never reformatted by Sheets.

## How sets are drawn

- **Split by image.** Test and train crops never come from the same image, and
  a set never mixes the two.
- **Test first.** Each annotator's first 150 answered sets are test; after
  that, new sets are test half the time. These are constants at the top of
  `Code.gs`.
- **Test sets are independent.** Each uses three never-used test crops from
  three different images. Train sets reuse train crops, least-used first.
- **Shared sets.** About 1 in 5 test sets goes to both annotators. Their
  agreement is the human ceiling.
- **Position is randomised** each time a set is shown.
- **Two people at once:** a script lock makes sure they're never given the
  same test crop.

## Scoring

In the Sheet, run **Triplets > Export for scoring**. It writes `crops.csv`,
`triplets.csv`, `annotations.csv` and `retractions.csv` into an `exports`
folder next to the crops. Copy that folder somewhere, then:

```bash
python motifs/expert_triplets/score_triplets.py --data-dir <exports folder> \
    --motifs <root>/stage4/motifs.parquet --embeddings <root>/stage3/embeddings.parquet
```

It reports, for the test split:

- **Motif agreement.** Credit 1 if the expert's pair shares a motif and the odd
  crop doesn't, 0 if the motifs contradict the expert, 1/3 if the motifs have no
  opinion. Chance is exactly 1/3, plus a permutation null.
- **Human ceiling** and the **normalised score**
  (score − 1/3) / (ceiling − 1/3), shown once there are 30+ shared sets.
- **Embedding accuracy and odd-one-out loss.** Uniform guessing gives a loss of
  1.0986; lower is better.
- **Anytime 95% intervals**, which stay valid however often you check while
  annotation continues.

**Training:** `triplet_metrics.torch_odd_one_out_loss` is the same loss in
trainable form, for fitting per-dimension weights on the frozen embeddings
using train answers only. Pass the weights back with `--weights` to compare
test scores before and after.
