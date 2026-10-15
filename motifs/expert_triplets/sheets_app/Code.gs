/**
 * Odd-one-out annotation app — Google Apps Script backend, bound to a Sheet.
 *
 * Always-on and free: Google hosts the page, the Sheet stores everything, and
 * the crop images stay private in your Drive (this script, running as you,
 * reads them and sends them to the page; nothing is shared publicly).
 *
 * Tabs (created by "Triplets > 1. Set up"):
 *   crops        imported from crops.csv, plus each image's Drive file id
 *   triplets     every triplet ever created (append-only)
 *   annotations  every answer that currently counts
 *   retractions  undone answers: undo moves the full row here, so nothing is
 *                ever lost and `annotations` shows exactly what counts
 * Motif labels are never stored here, so the log stays blind and scores any
 * model run. "Triplets > Export for scoring" writes the tabs as CSVs for
 * score_triplets.py.
 *
 * Setup and deployment steps are in README.md next to this file.
 */

const APP_VERSION = 'triplets-sheets-v5';
const INSTRUCTIONS_VERSION = 'v2';
// How triplets are drawn (same defaults as the design notes):
const TEST_FIRST = 150;    // answered test triplets per annotator before mixing
const TEST_SHARE = 0.5;    // after that, share of new triplets that are test
const SHARED_EVERY = 5;    // about 1 in N test triplets goes to every annotator

const HEADERS = {
  triplets: ['triplet_id', 'created_at', 'created_for', 'split', 'shared',
             'crop_1', 'crop_2', 'crop_3'],
  annotations: ['annotation_id', 'triplet_id', 'annotator', 'shown_left',
                'shown_middle', 'shown_right', 'choice', 'odd_crop',
                'response_ms', 'shown_at', 'answered_at', 'app_version',
                'instructions_version', 'render_version', 'unusable_crops'],
};
HEADERS.retractions = ['retracted_at'].concat(HEADERS.annotations);
const POSITIONS = ['left', 'middle', 'right'];

// ------------------------------------------------------------ properties --
function prop_(k) { return PropertiesService.getScriptProperties().getProperty(k); }
function setProp_(k, v) { PropertiesService.getScriptProperties().setProperty(k, v); }
function now_() { return new Date().toISOString(); }

// ------------------------------------------------------------- the menu ---
function onOpen() {
  SpreadsheetApp.getUi().createMenu('Triplets')
    .addItem('1. Set up', 'menuSetup')
    .addItem('2. Import crops from Drive', 'menuImportCrops')
    .addSeparator()
    .addItem('Export for scoring', 'menuExport')
    .addItem('Show settings', 'menuInfo')
    .addToUi();
}

function menuSetup() {
  const ui = SpreadsheetApp.getUi();
  const folder = ui.prompt('Set up (1/2)',
    'Paste the link to the Drive folder that holds crops.csv, ' +
    'render_config.json and the crops/ folder:', ui.ButtonSet.OK_CANCEL);
  if (folder.getSelectedButton() !== ui.Button.OK) return;
  const id = (folder.getResponseText().match(/[-\w]{25,}/) || [])[0];
  if (!id) { ui.alert('That does not look like a Drive folder link.'); return; }
  DriveApp.getFolderById(id);                       // fails loudly if no access
  const names = ui.prompt('Set up (2/2)',
    'Annotator names for the start screen, comma-separated:',
    ui.ButtonSet.OK_CANCEL);
  if (names.getSelectedButton() !== ui.Button.OK) return;

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  setProp_('SHEET_ID', ss.getId());
  setProp_('FOLDER_ID', id);
  setProp_('ANNOTATORS', names.getResponseText() || 'Prof. Huang, Carver');
  if (!prop_('TOKEN')) setProp_('TOKEN', Utilities.getUuid().replace(/-/g, ''));
  for (const name of ['crops', 'triplets', 'annotations', 'retractions']) {
    if (!ss.getSheetByName(name)) ss.insertSheet(name);
    const sh = ss.getSheetByName(name);
    if (HEADERS[name]) ensureHeader_(sh, HEADERS[name]);
  }
  ui.alert('Set up done. Next: Triplets > 2. Import crops from Drive, then ' +
           'deploy the web app (README step 5).\n\nAccess key for the link: ' +
           prop_('TOKEN'));
}

function menuImportCrops() {
  const ui = SpreadsheetApp.getUi();
  const res = importCrops_();
  ui.alert(`Imported ${res.added} new crop(s) (${res.byText}).` +
           (res.missing ? `\n\n${res.missing} row(s) skipped: image file not ` +
                          `found in the crops/ folder (upload incomplete?).` : '') +
           `\n\nThe sheet now holds ${res.total} crop(s).`);
}

function menuExport() {
  const url = exportForScoring_();
  SpreadsheetApp.getUi().alert('Wrote crops.csv, triplets.csv, annotations.csv ' +
    'and retractions.csv to the "exports" folder:\n' + url);
}

function menuInfo() {
  SpreadsheetApp.getUi().alert(
    `Access key: ${prop_('TOKEN')}\nAnnotators: ${prop_('ANNOTATORS')}\n` +
    `Render version: ${prop_('RENDER_VERSION')}\n\n` +
    'Share: <web app URL>?token=<access key>');
}

// ------------------------------------------------------ crops and export --
function importCrops_() {
  const folder = DriveApp.getFolderById(prop_('FOLDER_ID'));
  const csvFile = folder.getFilesByName('crops.csv');
  if (!csvFile.hasNext()) throw new Error('crops.csv not found in the folder');
  const rows = Utilities.parseCsv(csvFile.next().getBlob().getDataAsString());
  const head = rows.shift();
  const cfgFile = folder.getFilesByName('render_config.json');
  if (cfgFile.hasNext()) {
    const rv = JSON.parse(cfgFile.next().getBlob().getDataAsString()).render_version;
    if (prop_('RENDER_VERSION') && prop_('RENDER_VERSION') !== rv)
      throw new Error(`render version changed (${prop_('RENDER_VERSION')} -> ${rv}); ` +
                      'crops must keep one scale');
    setProp_('RENDER_VERSION', rv);
  }

  // Drive file id for every image under crops/test and crops/train
  const ids = {};
  const cropsDir = folder.getFoldersByName('crops');
  if (!cropsDir.hasNext()) throw new Error('crops/ folder not found');
  const sub = cropsDir.next().getFolders();
  while (sub.hasNext()) {
    const f = sub.next();
    const files = f.getFiles();
    while (files.hasNext()) {
      const file = files.next();
      ids[f.getName() + '/' + file.getName()] = file.getId();
    }
  }

  const sh = sheet_('crops');
  const have = new Set(readTable_('crops').map(r => r.crop_id));
  if (sh.getLastRow() === 0) appendRows_(sh, [head.concat(['drive_file_id'])]);
  const iCrop = head.indexOf('crop_id'), iFile = head.indexOf('file'),
        iSplit = head.indexOf('split');
  const add = [], count = {};
  let missing = 0;
  for (const r of rows) {
    if (!r[iCrop] || have.has(r[iCrop])) continue;
    const key = r[iFile].split('/').slice(-2).join('/');     // test/<name>
    if (!ids[key]) { missing++; continue; }
    add.push(r.concat([ids[key]]));
    count[r[iSplit]] = (count[r[iSplit]] || 0) + 1;
  }
  if (add.length) appendRows_(sh, add);
  return { added: add.length, missing: missing, total: have.size + add.length,
           byText: Object.entries(count).map(([k, v]) => `${v} ${k}`).join(', ') || 'none' };
}

function exportForScoring_() {
  const folder = DriveApp.getFolderById(prop_('FOLDER_ID'));
  const it = folder.getFoldersByName('exports');
  const out = it.hasNext() ? it.next() : folder.createFolder('exports');
  for (const name of ['crops', 'triplets', 'annotations', 'retractions']) {
    const values = sheet_(name).getDataRange().getDisplayValues();
    const csv = values.map(r => r.map(csvCell_).join(',')).join('\n') + '\n';
    const f = out.getFilesByName(name + '.csv');
    if (f.hasNext()) f.next().setContent(csv);
    else out.createFile(name + '.csv', csv, MimeType.CSV);
  }
  return out.getUrl();
}

function csvCell_(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// ------------------------------------------------------------ the sheet ---
function sheet_(name) {
  return SpreadsheetApp.openById(prop_('SHEET_ID')).getSheetByName(name);
}

/** Rows as objects keyed by the header row; every value read back as text. */
function readTable_(name) {
  const values = sheet_(name).getDataRange().getDisplayValues();
  if (values.length < 2) return [];
  const head = values[0];
  return values.slice(1).filter(r => r.some(v => v !== '')).map(r => {
    const o = {};
    head.forEach((h, i) => { o[h] = r[i]; });
    return o;
  });
}

/** Append as plain text, so ids like "001" and ISO timestamps are never
 *  turned into numbers or dates by Sheets. */
function appendRows_(sh, rows) {
  const start = sh.getLastRow() + 1;
  const width = Math.max(...rows.map(r => r.length));
  const padded = rows.map(r => r.concat(Array(width - r.length).fill('')).map(String));
  sh.getRange(start, 1, padded.length, width).setNumberFormat('@').setValues(padded);
}

/** Write `head` as row 1 of an empty tab (or one holding only an outdated
 *  header). Refuses to touch a tab whose rows use another layout. */
function ensureHeader_(sh, head) {
  const n = sh.getLastRow();
  if (n === 0 || n === 1) {
    if (n === 1 && sh.getLastColumn() > head.length)
      sh.getRange(1, 1, 1, sh.getLastColumn()).clearContent();
    sh.getRange(1, 1, 1, head.length).setNumberFormat('@').setValues([head]);
    return;
  }
  const cur = sh.getRange(1, 1, 1, head.length).getDisplayValues()[0];
  if (cur.join() !== head.join())
    throw new Error(`The "${sh.getName()}" tab has rows in an older layout. ` +
                    'Delete every row below its header, then try again.');
}

function withLock_(fn) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try { return fn(); } finally { lock.releaseLock(); }
}

function liveAnswers_() {
  const gone = new Set(readTable_('retractions').map(r => r.annotation_id));
  return readTable_('annotations').filter(a => !gone.has(a.annotation_id));
}

/** Crops anyone has flagged unusable (in an answer that wasn't undone). */
function flaggedCrops_(live) {
  const out = new Set();
  live.forEach(a => String(a.unusable_crops || '').split(';').filter(Boolean)
                      .forEach(c => out.add(c)));
  return out;
}

function shuffle_(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** The page's copy of a set: crop order on screen plus the images, read from
 *  Drive. Always called OUTSIDE the lock: reading images is the slow part, and
 *  it must not hold up other requests. */
function payload_(t, crops, order) {
  const byId = {};
  crops.forEach(c => { byId[c.crop_id] = c; });
  const ids = [t.crop_1, t.crop_2, t.crop_3];
  const shown = order || shuffle_(ids.slice());              // random position
  const images = shown.map(cid => {
    const blob = DriveApp.getFileById(byId[cid].drive_file_id).getBlob();
    return 'data:' + blob.getContentType() + ';base64,' +
           Utilities.base64Encode(blob.getBytes());
  });
  return { triplet_id: t.triplet_id, order: shown, images: images };
}

/** Three crops of `split` from three different images. Test crops are used
 *  once ever; train crops are reused, least-used first. */
function pick_(split, crops, triplets, flagged) {
  const used = {};
  triplets.forEach(t => [t.crop_1, t.crop_2, t.crop_3].forEach(c => {
    used[c] = (used[c] || 0) + 1;
  }));
  let cands = shuffle_(crops.filter(c => c.split === split && !flagged.has(c.crop_id) &&
                                         (split !== 'test' || !used[c.crop_id])));
  if (split === 'train') cands.sort((a, b) => (used[a.crop_id] || 0) - (used[b.crop_id] || 0));
  const chosen = [], images = new Set();
  for (const c of cands) {
    const img = c.condition + '|' + c.image_id;
    if (images.has(img)) continue;
    chosen.push(c.crop_id);
    images.add(img);
    if (chosen.length === 3) return chosen;
  }
  return null;
}

// --------------------------------------------------------- the web app ----
function doGet(e) {
  if (!prop_('TOKEN') || (e.parameter.token || '') !== prop_('TOKEN')) {
    return HtmlService.createHtmlOutput(
      '<p style="font-family:sans-serif">This link is missing its access key. ' +
      'Please use the full link you were sent.</p>');
  }
  const t = HtmlService.createTemplateFromFile('Index');
  t.token = prop_('TOKEN');
  return t.evaluate().setTitle('Odd One Out')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function checked_(token, who) {
  if (token !== prop_('TOKEN')) throw new Error('missing or wrong access key');
  if (who === undefined) return '';
  const w = String(who || '').trim().slice(0, 80);
  if (!w) throw new Error('no annotator name');
  return w;
}

function apiConfig(token) {
  checked_(token);
  return { annotators: (prop_('ANNOTATORS') || '').split(',').map(s => s.trim()).filter(Boolean),
           instructions_version: INSTRUCTIONS_VERSION };
}

/** Everything one request needs from the Sheet, read once. The helpers below
 *  update it in place, so later steps in the same request see earlier ones. */
function state_() {
  return { crops: readTable_('crops'), triplets: readTable_('triplets'), live: liveAnswers_() };
}

function progress_(who, live) {
  const mine = live.filter(a => a.annotator === who);
  const today = new Date().toISOString().slice(0, 10);
  return { total: mine.length, today: mine.filter(a => a.answered_at >= today).length };
}

/** Choose the next set for `who` (call inside the lock). Sets already handed
 *  out but unanswered come first; otherwise a new one is created. `skip` holds
 *  the sets the page already has. Returns the triplet row, or null if none. */
function nextTriplet_(who, skip, st, nAnnotators) {
  const mine = new Set(st.live.filter(a => a.annotator === who).map(a => a.triplet_id));
  const pending = st.triplets
    .filter(t => (t.created_for === who || (t.shared === '1' && t.created_for !== who)) &&
                 !mine.has(t.triplet_id) && !skip.has(t.triplet_id))
    .sort((a, b) => (b.created_for === who) - (a.created_for === who) ||
                    (a.created_at < b.created_at ? -1 : 1));
  if (pending.length) return pending[0];

  const split0 = {};
  st.triplets.forEach(t => { split0[t.triplet_id] = t.split; });
  const nTest = st.live.filter(a => a.annotator === who && split0[a.triplet_id] === 'test').length;
  let split = nTest < TEST_FIRST ? 'test' : (Math.random() < TEST_SHARE ? 'test' : 'train');
  const flagged = flaggedCrops_(st.live);
  let chosen = pick_(split, st.crops, st.triplets, flagged);
  if (!chosen && split === 'test') { split = 'train'; chosen = pick_('train', st.crops, st.triplets, flagged); }
  if (!chosen) return null;
  const shared = split === 'test' && nAnnotators > 1 && Math.random() < 1 / SHARED_EVERY;
  const t = { triplet_id: Utilities.getUuid().replace(/-/g, '').slice(0, 12),
              created_at: now_(), created_for: who, split: split,
              shared: shared ? '1' : '0',
              crop_1: chosen[0], crop_2: chosen[1], crop_3: chosen[2] };
  appendRows_(sheet_('triplets'), [HEADERS.triplets.map(h => t[h])]);
  st.triplets.push(t);
  return t;
}

/** Validate and append one answer (call inside the lock). */
function saveAnswer_(who, body, st) {
  const t = st.triplets.find(x => x.triplet_id === body.triplet_id);
  if (!t) return { error: 'unknown triplet' };
  const ids = [t.crop_1, t.crop_2, t.crop_3];
  const order = body.order || [];
  if (order.length !== 3 || order.slice().sort().join() !== ids.slice().sort().join())
    return { error: 'order does not match the triplet' };
  if (POSITIONS.concat(['unusable']).indexOf(body.choice) < 0) return { error: 'bad choice' };
  const unusable = Array.from(new Set(body.unusable || []));
  if (unusable.some(c => ids.indexOf(c) < 0)) return { error: 'flagged crop is not in this set' };
  if ((body.choice === 'unusable') !== (unusable.length > 0))
    return { error: 'unusable needs at least one flagged image, and a pick needs none' };
  if (t.created_for !== who && t.shared !== '1')
    return { error: 'triplet belongs to another annotator' };
  if (st.live.some(a => a.annotator === who && a.triplet_id === t.triplet_id))
    return { ok: true, duplicate: true };                       // resent or double-click
  const sh = sheet_('annotations');
  const id = Utilities.getUuid().replace(/-/g, '').slice(0, 12);
  const odd = body.choice === 'unusable' ? '' : order[POSITIONS.indexOf(body.choice)];
  const row = { annotation_id: id, triplet_id: t.triplet_id, annotator: who,
                shown_left: order[0], shown_middle: order[1], shown_right: order[2],
                choice: body.choice, odd_crop: odd,
                response_ms: String(Math.round(Number(body.response_ms) || 0) || ''),
                shown_at: String(body.shown_at || ''), answered_at: now_(),
                app_version: APP_VERSION, instructions_version: INSTRUCTIONS_VERSION,
                render_version: prop_('RENDER_VERSION') || '',
                unusable_crops: unusable.join(';') };
  appendRows_(sh, [HEADERS.annotations.map(h => row[h])]);
  st.live.push(row);
  return { ok: true, annotation_id: id };
}

/** Undo `who`'s answer to triplet `tid` (call inside the lock): delete its
 *  row from `annotations` and keep the full row in `retractions`. */
function retract_(who, tid, st) {
  const a = st.live.find(x => x.annotator === who && x.triplet_id === tid);
  if (!a) return;
  const sh = sheet_('annotations');
  const vals = sh.getDataRange().getDisplayValues();
  const col = vals[0].indexOf('annotation_id');
  const r = vals.findIndex((row, i) => i > 0 && row[col] === a.annotation_id);
  const rs = sheet_('retractions');
  ensureHeader_(rs, HEADERS.retractions);
  appendRows_(rs, [[now_()].concat(HEADERS.annotations.map(h => a[h] || ''))]);
  if (r > 0) sh.deleteRow(r + 1);
  st.live = st.live.filter(x => x !== a);
}

/** One round trip per step: apply any undos, save any answers the page is
 *  holding, then hand
 *  back up to `want` (max 3) more sets. Answers are deduplicated, so the page
 *  can safely resend after a failed request. Images are read after the lock
 *  is released. */
function apiSync(token, who, answers, exclude, want, retracts) {
  who = checked_(token, who);
  const nAnnotators = apiConfig(token).annotators.length;
  const r = withLock_(() => {
    const st = state_();
    // Undos first, so an undo followed by a new answer to the same set
    // (sent together) ends with the new answer counted.
    (retracts || []).forEach(tid => retract_(who, tid, st));
    const saved = [], failed = [];
    (answers || []).forEach(body => {
      const res = saveAnswer_(who, body, st);
      if (res.error) failed.push({ triplet_id: body.triplet_id, error: res.error });
      else saved.push(body.triplet_id);
    });
    const skip = new Set(exclude || []), picks = [];
    let empty = false;
    for (let i = 0; i < Math.min(Number(want) || 0, 3); i++) {
      const t = nextTriplet_(who, skip, st, nAnnotators);
      if (!t) { empty = true; break; }
      skip.add(t.triplet_id);
      picks.push(t);
    }
    return { saved, failed, empty, picks, crops: st.crops, progress: progress_(who, st.live) };
  });
  return { saved: r.saved, failed: r.failed, retracted: retracts || [], empty: r.empty,
           progress: r.progress,
           sets: r.picks.map(t => payload_(t, r.crops)) };
}

/** Single-set and single-answer forms of apiSync. */
function apiNext(token, who, exclude) {
  const r = apiSync(token, who, [], exclude, 1);
  return r.sets[0] || { empty: true };
}

function apiAnswer(token, body) {
  const who = checked_(token, body && body.annotator);
  return withLock_(() => {
    const st = state_();
    const res = saveAnswer_(who, body, st);
    return res.error ? res : Object.assign(res, { progress: progress_(who, st.live) });
  });
}

function apiProgress(token, who) {
  who = checked_(token, who);
  const mine = liveAnswers_().filter(a => a.annotator === who);
  const today = new Date().toISOString().slice(0, 10);
  return { total: mine.length, today: mine.filter(a => a.answered_at >= today).length };
}
