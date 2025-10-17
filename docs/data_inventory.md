# Dataset Inventory & Integration Notes

Catalogue of locally available datasets, reference links, and key integration notes derived from source documentation.

| Dataset | Local Path | Source URL | Licence | Schema Highlights | Integration Notes |
|---------|------------|------------|---------|-------------------|-------------------|
| Balabit Mouse Dynamics Challenge | `datasets/Mouse-Dynamics-Challenge-master` | https://github.com/balabit/Mouse-Dynamics-Challenge | No explicit licence; cite repository | Session CSV columns: `record timestamp`, `client timestamp`, `button`, `state`, `x`, `y`. Training/test split by user folders; `public_labels.csv` annotates subset of test. | Remote desktop capture; timestamps relative to session start. Test folders mix other users for attack simulation—keep folder user IDs. Buttons encoded as bitmask. Citation: Fülöp, Á.; Kovács, L.; Kurics, T.; Windhager-Pokol, E. (2016). |
| Boğaziçi University Mouse Dynamics | `datasets/boun-mouse-dynamics-dataset` | https://doi.org/10.17632/w6cxr8yc7p.1 | CC BY 4.0 | Seven variables captured by continuous logger; per-user directories with training, internal, external subsets. | Columns include client timestamp, coordinates, button, state, window title. Loader in `src/data/bogazici_loader.py`. Raw archive checksum recorded in `data/raw/bogazici/checksums.txt` (sha256 `5684d5b89e24a850e663a94224f46090b696b9983efa3620b82fca6cb3291c5d`). Citation: Yıldırım, M.; Kılıç, A. A.; Anarım, E. (2020). |
| SapiMouse | `datasets/sapimouse-main` | https://github.com/sapimouse/sapimouse | No explicit licence; cite repository/paper | Raw logs ideally `[timestamp, button, state, x, y]` but this repo ships processed ABS Δx/Δy blocks. | Current loader (`src/data/sapimouse_loader.py`) handles processed CSVs as provisional input (emits synthetic timestamps). Replace with raw log parsing once official data added. Citation: Antal, M.; Fejer, N.; Buza, K. (2021). |
| Attentive Cursor | `datasets/the-attentive-cursor-dataset-master` | https://gitlab.com/iarapakis/the-attentive-cursor-dataset | CC BY-NC-SA 4.0 | Event CSV columns: `cursor`, `timestamp`, `xpos`, `ypos`, `event`, `xpath`, `attrs`, `extras`. XML metadata per log. | Loader (`src/data/attentive_loader.py`) reads CSV logs, filters to mouse events, merges ground-truth + participant metadata. Citation: Leiva, L. A.; Arapakis, I. (2020). |

## Citations
- Balabit Mouse Dynamics Challenge — Fülöp, Á.; Kovács, L.; Kurics, T.; Windhager-Pokol, E. (2016). *Balabit Mouse Dynamics Challenge data set*. https://github.com/balabit/Mouse-Dynamics-Challenge.
- Boğaziçi University Mouse Dynamics Dataset — Yıldırım, M.; Kılıç, A. A.; Anarım, E. (2020). *Boğaziçi University Mouse Dynamics Dataset*. Mendeley Data, V1. doi:10.17632/w6cxr8yc7p.1.
- SapiMouse Dataset — Antal, M.; Fejer, N.; Buza, K. (2021). *SapiMouse: Mouse Dynamics-based User Authentication Using Deep Feature Learning*. IEEE Access. Repository: https://github.com/sapimouse/sapimouse.
- Attentive Cursor Dataset — Leiva, L. A.; Arapakis, I. (2020). *The Attentive Cursor Dataset*. Front. Hum. Neurosci. 14. doi:10.3389/fnhum.2020.565664.

## Next Actions
- Monitor Balabit and SapiMouse repositories for licence updates; continue citing sources directly.
- Replace provisional SapiMouse loader once raw logs are available.
- Record additional checksums when archiving other datasets.
- Persist preprocessing outputs under `data/processed/` with metadata manifests.

