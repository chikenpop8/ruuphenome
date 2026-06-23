# H100 Training — RuuPhenome SSL Encoder

Ready-to-submit guide for retraining the self-supervised masked NMR encoder on
the KKU H100 cluster (LiCO). Open-data only — **can run from June 26** (no
hackathon dataset needed). Optional: a stronger encoder → better spectral
embeddings for clustering/visualization. The prototype works without it.

---

## Before you start
1. Connect to the **KKU VPN** (credentials arrive June 26 via Discord).
2. Open the LiCO portal: <https://10.198.253.15:8000> → accept the self-signed
   cert → log in.
3. Upload the project (or `git clone` it) into your LiCO workspace.

---

## Submit the job (LiCO → Job Templates → Common Job)

| Field | Value |
|---|---|
| Job Name | `ruuphenome-train` |
| Workspace | your project directory |
| Queue | `GPU_FOR_BDI` |
| Node | `1` |
| CPU Cores Per Node | `2` |
| GPU Per Node | `1` |
| GPU Resource Type | `gpu:STU_GPU` |
| Memory Used | `8192` (8 GB) |
| Wall Time | `01:00:00` (1 h is ample) |

### Run Script

```bash
# install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.bashrc"

cd ruuphenome/backend            # <-- adjust to your uploaded path

# create env + install deps
uv venv .venv && . .venv/bin/activate
uv pip install -r nmr_api/requirements.txt

# retrain the encoder (H100-sized run)
python -m nmr_api.train_on_h100 \
  --epochs 200 --steps-per-epoch 128 \
  --batch-size 256 --embedding-dim 128
```

---

## What it does
1. Ensures the open **BMRB** reference corpus exists (downloads if missing).
2. Retrains the masked-autoencoder encoder with H100-scale settings.
3. Runs the augmented-retrieval benchmark on the new encoder.
4. Writes:
   - `nmr_api/models/masked_nmr_encoder.pt` — new checkpoint
   - `nmr_api/models/masked_nmr_training.json` — loss history + config
   - `nmr_api/models/h100_training_report.json` — full report (device, timing, benchmark)

## After the job
- **Job Monitoring → job name → Log** shows the live output and the final report.
- Download the new `masked_nmr_encoder.pt` and commit it, or copy it onto the VM.
- Restart the server (`bash run.sh`) to load the retrained encoder.

## Honest framing for the pitch
> "We retrained our representation model on the H100 cluster using open BMRB
> reference data — something a closed tool like Chenomx cannot do."

This is a *representation/visualization* improvement. Metabolite **annotation**
(Track 1) and **biomarker discovery** (Track 2) do **not** require this job —
they run on CPU in seconds. Only run this if you want the stronger-embedding story.
