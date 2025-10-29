# Local Mouse Gesture Collector

Use this guide to record high-resolution mouse dynamics that align with our Boğaziçi-based preprocessing stack. The collector emits raw session logs, immediate gesture segmentation, neuromotor features, and octant (+stationary) buckets so the dataset is ready for training without additional transforms.

## Why This Exists
- Boğaziçi provides ~40 Hz traces; modern gaming mice ship ≥1 kHz updates. Capturing at 200 Hz+ closes that gap so the generator and discriminator can specialise on your hardware.
- Produces consistent `NormalisedEvent` records (`client_timestamp`, `x`, `y`, `button`, `state`) while augmenting with deltas, inactivity-driven segment boundaries, and bucket metadata.
- Session boundaries (5 minutes idle) and gesture gaps (1 second idle) match our segmentation heuristics and allow session-level replay analyses.
- Includes jitter isolation (short path-length desk bumps) for optional specialised training.

## Quick Start
```bash
python3 scripts/collect_local_mouse.py \
  --user-id your_id \
  --dataset-id local_mouse \
  --sampling-hz 200 \
  --gesture-idle 1 \
  --session-idle 300 \
  --sequence-length 64
```

Press `Ctrl+C` to stop. The collector logs progress and writes manifests as soon as a session closes.

Add `--ui` to launch a lightweight Tkinter dashboard that lets you start/stop capture and tag each session without touching the terminal. Pick a session type (required before the `Start` button enables), optionally add notes, then hit `Start` to open a new session; `Stop` finalises it. The UI also surfaces the currently focused desktop application (detected automatically via `xdotool` on Linux) so you can sanity-check that gestures are being bucketed under the right process/site.

### CLI Flags of Note
- `--sampling-hz`: polling rate for position (default 200 Hz). Adjust up if your mouse reports faster; down if CPU spikes.
- `--skip-stationary-duplicates`: drop identical samples that arrive faster than the display refresh to reduce redundancy.
- `--no-processed`: disable on-the-fly segmentation/feature export if resources are constrained (raw CSV only).
- `--no-features`: keep segmentation but skip neuromotor feature computation (faster, smaller files).
- `--no-stationary-bucket`: revert to 8 pure octants if the +1 stationary bucket is not desired.
- `--no-click-split`: rely solely on inactivity gaps (default splits on each button press).
- `--ui`: open the basic session-label UI instead of the blocking CLI loop.
- `--session-types`: override the preset session-type dropdown entries exposed in the UI (one of these must be chosen before `Start`).
- `--default-session-type`, `--default-notes`: seed the initial labels applied to each gesture/session.
- `--linked-gap-seconds`: threshold that links successive gestures into a sequence when the idle gap ≤ this value (default 0.2 s).

See `python3 scripts/collect_local_mouse.py --help` for the complete list.

## Data Layout
```
data/
├── raw/
│   └── local_mouse/
│       └── {session_id}/
│           ├── events.csv
│           └── session_manifest.json
└── processed/
    └── local_mouse/
        ├── bucket_00_W/
        ├── …
        ├── bucket_07_SW/
        ├── bucket_08_Stationary/   # if enabled
        └── jitter/
```

### `events.csv`
- Columns: `client_timestamp`, `x`, `y`, `button`, `state`, `source`, `delta_x`, `delta_y`, `delta_t`, `application`, `window_title`, `process_name`, `process_id`.
- `source` is `listener` (OS event) or `poll` (fixed-rate sampler).
- Timestamps are UNIX seconds (float). Button/state mirror Boğaziçi (`Move`, `Pressed`, `Released`, `Scroll`).

### `session_manifest.json`
```json
{
  "session_id": "1729796910_ab12cd34",
  "user_id": "local_user",
  "dataset_id": "local_mouse",
  "split": "train",
  "gesture_count": 154,
  "median_sample_rate_hz": 198.6,
  "duration_seconds": 472.1,
  "session_start_timestamp": 1729796910.002,
  "session_context": {
    "session_type": "poker",
    "notes": "evening run"
  },
  "context_changes": [
    {
      "timestamp": 1729796910.002,
      "reason": "session_start",
      "context": {
        "session_type": "poker",
        "notes": "evening run"
      }
    }
  ],
  "application_changes": [
    {
      "timestamp": 1729796910.115,
      "reason": "application_change",
      "application": {
        "app_id": "pokerstars",
        "process_name": "pokerstars",
        "window_title": "PokerStars Lobby",
        "pid": "94712"
      }
    }
  ],
  "application_usage": [
    {"application": "pokerstars", "count": 7423}
  ],
  "active_application": {
    "app_id": "firefox",
    "process_name": "firefox",
    "window_title": "Break - Mozilla Firefox",
    "pid": "95331"
  },
  "gestures": [
    {
      "gesture_id": "1729796910_ab12cd34_g00000",
      "bucket": 4,
      "bucket_label": "E",
      "duration_seconds": 0.42,
      "path_length": 314.7,
      "event_count": 87,
      "is_jitter": false,
      "start_timestamp": 1729796911.021,
      "end_timestamp": 1729796911.441,
      "session_context": {
        "session_type": "poker",
        "notes": "evening run"
      },
      "application_info": {
        "application": "pokerstars",
        "window_title": "PokerStars Lobby",
        "process_name": "pokerstars",
        "process_id": 94712
      },
      "application_counts": [
        {"application": "pokerstars", "count": 87}
      ],
      "link_group_id": "1729796910_ab12cd34_lg0001",
      "link_group_sequence": 0,
      "linked_previous_gesture_id": null,
      "linked_previous_gap_seconds": null,
      "linked_next_gesture_id": "1729796910_ab12cd34_g00001",
      "linked_next_gap_seconds": 0.18,
      "inter_gesture_gap_seconds": null,
      "gesture_category": "linked",
      "npz_path": "bucket_04_E/1729796910_ab12cd34_g00000.npz"
    },
    …
  ]
}
```

### Processed Gesture Files (`.npz`)
- Stored per bucket: `bucket_{index:02d}_{label}/{gesture_id}.npz`.
- Arrays:
  - `sequence`: `(64, 3)` tensor of `(Δx, Δy, Δt)` after interpolation at `sampling_hz`.
  - `mask`: valid timestep mask.
  - `neuromotor_features`: optional `(F,)` float vector (`compute_features=True`).
  - `metadata_json`: JSON blob with dataset identifiers, path length, duration, jitter flag, `session_context` (session labels), `application_info`/`application_counts` (dominant foreground process during the gesture + per-app counts), and link metadata (`link_group_id`, `link_group_sequence`, `linked_previous/next_*`, `inter_gesture_gap_seconds`, `gesture_category`).
- A copy is placed under `jitter/` whenever `path_length ≤ jitter_path_threshold`.

## Segmentation & Bucketing Rules
- **Gesture boundary**: finalise when idle for `gesture_idle_seconds` (default 1 s), on any button press (configurable), or on session rollover.
- **Session boundary**: idle ≥ `session_idle_seconds` (default 5 min) -> new folder, fresh manifest.
- **Direction buckets**: eight octants derived from total `(Δx, Δy)` using the same atan2 logic as `GestureDataset`; optional bucket 8 for stationary gestures (`‖Δ‖ ≤ stationary_threshold`).
- **Jitter detection**: `path_length ≤ jitter_path_threshold` (pixels) marks the gesture for the jitter replay pool.
- **Sample rate estimate**: per-session median `Δt` from captured samples; stored in manifest for post-hoc reconfiguration.
- **Linked gestures**: when the idle gap between successive gestures ≤ `linked_gesture_gap_seconds` (default 0.2 s), both gestures receive a shared `link_group_id` and reciprocal `linked_*` metadata so short click+move chains can be replayed as organic sequences.

## Integration With Training Pipeline
- Dataset registry entry: `dataset_id="local_mouse"` mapped to `data/raw/local_mouse`.
- Loader: `data.local_mouse_collector.load_local_mouse` yields `NormalisedEvent` records ready for `GestureDataset`.
- Example Hydra override:
```yaml
data:
  dataset_id: local_mouse
  sampling_rate: 200.0
  sequence_length: 64
  min_events: 5
  direction_buckets: [0,1,2,3,4,5,6,7,8]  # include stationary bucket
  rotate_to_buckets: false
```
- For jitter-only experiments point the dataloader at `data/processed/local_mouse/jitter`.

## Operational Notes
- Requires `pynput>=1.7`; install via `.venv/bin/pip install -r requirements.txt` and regenerate `requirements.lock`.
- Foreground application tracking relies on `xdotool` on Linux (`sudo apt install xdotool`); without it the application metadata fields remain empty.
- Works cross-platform (Windows/macOS/Linux). On Wayland, extra permissions may be needed for global pointer hooks.
- CPU impact: at 200 Hz polling the collector uses <3 % CPU on recent laptops; adjust `--sampling-hz` if resources drop.
- Privacy: captured data never leaves `data/raw/local_mouse`; do not sync this directory unless anonymised.

## Next Steps / Ideas
- Add optional window title capture (requires OS-specific APIs).
- Wire jitter bucket directly into replay buffer sampling.
- Surface a small dashboard summarising octant distribution vs Boğaziçi baseline.
