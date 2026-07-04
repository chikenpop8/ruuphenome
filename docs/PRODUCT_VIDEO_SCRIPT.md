# RuuPhenome — Product Motion Graphic Script

**Runtime: 1:30 (90s) · Format: SaaS product tour · No "Track 1 / Track 2" language, no problem-statement cold open**

Functions are named the way a user would describe them, not the way the codebase names them:

| On-screen function | What it actually is (for reference) |
|---|---|
| Upload | Binned ¹H matrix ingestion, auto-orientation, auto-profiling |
| Identify | Reference-shift matching, condition-aware (solvent/D₂O/matrix) |
| Quantify | NNLS deconvolution, per-compound fit quality |
| Confirm | Target–decoy FDR gating |
| Discover | Leakage-safe nested-CV biomarker panel discovery |
| Interpret | Pathway enrichment + curated biology + disease relevance |
| Export | Reproducible report / CSV |

---

## Scene 1 — Cold open (0:00–0:08 · 8s)

**Visual:** Black frame. A thin noisy baseline sweeps in left→right. A peak spikes; as it passes, the wordmark **RuuPhenome** resolves out of the waveform (mask-wipe synced to the spike). Subhead fades in beneath.

**On-screen text:** `RuuPhenome` → `¹H-NMR metabolomics, automated`

**VO:** *"RuuPhenome turns raw NMR spectra into metabolic insight — automatically."*

No preamble, no healthcare-problem framing — straight to product + one-line positioning.

---

## Scene 2 — Upload (0:08–0:20 · 12s)

**Visual:** A spectrum file drops into a minimal upload card. It unfolds into a live ppm-binned trace. Small stat chips pop in beside it: sample count, bin count, ppm range, a green "quality checked" tick.

**On-screen text:** `Upload` / *Any binned ¹H spectrum — auto-profiled and quality-checked on arrival.*

**VO:** *"Start with a binned spectrum from serum, plasma, or urine. RuuPhenome auto-detects the layout, profiles it, and flags anything unusual — before analysis even begins."*

---

## Scene 3 — Identify (0:20–0:34 · 14s)

**Visual:** Peaks along the trace light up one at a time; a name pill rises off each (Glucose, Lactate, Alanine, Citrate…). A small "solvent" chip toggles in the corner (aqueous ⇄ D₂O), signaling condition-awareness.

**On-screen text:** `Identify` / *Matched against a reference library — condition-aware, so calls stay honest.*

**VO:** *"Every peak is matched against a reference library of metabolites — accounting for solvent and sample type along the way, so identifications hold up."*

---

## Scene 4 — Quantify & Confirm (0:34–0:48 · 14s)

**Visual:** Translucent colored peak-areas stack under the black observed trace (compound-by-compound fit). A concentration list fills in beside it; each row gets a fit-quality dot, then a green ✓ as it clears confirmation.

**On-screen text:** `Quantify` → `Confirm` / *Overlapping peaks un-mixed into real concentrations — each one statistically confirmed.*

**VO:** *"Overlapping signals are mathematically un-mixed to measure how much of each compound is really there — with a confidence score on every call, and a statistical filter that keeps only what's confirmed."*

---

## Scene 5 — Discover (0:48–1:02 · 14s)

**Visual:** The concentration list collapses into a small bar panel (a biomarker set). An ROC curve draws itself top-right, AUC ticking upward. A grid of cross-validation folds flickers along the bottom, reinforcing "tested, not just fit."

**On-screen text:** `Discover` / *The smallest panel that tells your groups apart — validated to resist false discovery.*

**VO:** *"Add sample labels, and RuuPhenome searches for the smallest metabolite panel that separates your groups — cross-validated to resist false discovery, not just chase a good-looking number."*

---

## Scene 6 — Interpret (1:02–1:16 · 14s)

**Visual:** The bars morph into circular nodes; edges draw between related metabolites, forming a small pathway network. Colored relevance chips animate in beside nodes.

**On-screen text:** `Interpret` / *Pathways, biological roles, and disease relevance — automatically.*

**VO:** *"Every panel is placed in biological context automatically — pathway involvement, known roles, relevance across common conditions. A list of names becomes a story."*

---

## Scene 7 — Export (1:16–1:24 · 8s)

**Visual:** All prior elements shrink and converge into a single report card sliding out, with a download arrow and a short checklist ticking off (identifications, concentrations, biomarkers).

**On-screen text:** `Export` / *One reproducible report, ready for what's next.*

**VO:** *"The full analysis exports in one click — identifications, concentrations, and biomarkers together — ready for your next experiment."*

---

## Scene 8 — Close (1:24–1:30 · 6s)

**Visual:** Wordmark returns center-frame; tagline settles beneath; a small "open-source" mark appears last.

**On-screen text:** `RuuPhenome` / *From spectrum to insight.* / `open-source`

**VO:** *"RuuPhenome — from spectrum to insight."*

---

## Production notes

- **Palette:** near-black ground, off-white ink, electric blue (identify/data), acid lime (confirm/success), lavender (biology/interpret) — the product's own visual identity, not a stock gradient.
- **Type:** Neue Montreal (display) + a monospace for data/timecodes — same pairing as the live app, for brand continuity.
- **Transition motif:** a thin spectral pulse line runs continuously along the bottom edge of every scene and spikes at each cut — ties the video's own transitions to the subject matter instead of a generic wipe/blur.
- **Pacing:** 8 scenes, hard cuts on beat, ~10–14s each. No scene explains *why NMR matters* — every scene shows the product doing something.
- **Companion asset:** an autoplaying HTML motion-graphic implementing this script beat-for-beat is available as an Artifact — screen-record it for an actual video file, or use this script as a shot list for a designer / AI video tool + voiceover pass.
