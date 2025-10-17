from itertools import islice
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.attentive_loader import load_attentive_cursor
from data.balabit_loader import load_balabit
from data.bogazici_loader import load_bogazici
from data.sapimouse_loader import load_sapimouse


def take_events(loader, *args, **kwargs):
    return list(islice(loader(*args, **kwargs), 5))


def test_balabit_loader_produces_events():
    events = take_events(load_balabit, "train")
    assert events
    event = events[0]
    assert event.dataset_id == "balabit"
    assert event.user_id.startswith("user")
    assert event.timestamp_ms >= 0


def test_bogazici_loader_train_events():
    events = take_events(load_bogazici, "train")
    assert events
    event = events[0]
    assert event.dataset_id == "bogazici"
    assert event.split == "train"


def test_bogazici_loader_internal_events():
    events = take_events(load_bogazici, "internal")
    assert events
    assert events[0].split == "internal"


def test_sapimouse_loader_produces_rows():
    events = take_events(load_sapimouse)
    assert events
    assert events[0].dataset_id == "sapimouse"


def test_attentive_cursor_loader_mouse_events_only():
    events = take_events(load_attentive_cursor)
    assert events
    assert all(event.state.lower() in {"mousemove", "click", "mousedown", "mouseup", "wheel"} for event in events)

