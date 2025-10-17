from pathlib import Path
import sys
from itertools import islice
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data import (  # noqa: E402
    NormalisedEvent,
    events_to_gesture,
    load_balabit,
    segment_event_stream,
)


def collect_first_session_events(limit: int = 500) -> list[NormalisedEvent]:
    events: list[NormalisedEvent] = []
    first_key = None
    for event in load_balabit("train"):
        key = (event.user_id, event.session_id)
        if first_key is None:
            first_key = key
        if key != first_key or len(events) >= limit:
            break
        events.append(event)
    return events


def test_segment_event_stream_from_balabit():
    events = collect_first_session_events()
    gestures = list(
        segment_event_stream(
            events,
            gap_threshold_ms=250.0,
            target_len=32,
        )
    )
    assert gestures, "Expected at least one gesture from Balabit session"
    gesture = gestures[0]
    assert gesture.sequence.shape == (32, 3)
    assert np.isclose(gesture.mask, 1.0).all()
    assert gesture.duration > 0
    assert gesture.path_length >= 0


def synthetic_events(gap_ms: float, step: float = 1.0) -> Iterable[NormalisedEvent]:
    for idx in range(10):
        yield NormalisedEvent(
            dataset_id="synthetic",
            user_id="userA",
            session_id="session1",
            split="train",
            timestamp_ms=idx * 10.0,
            x=float(idx) * step,
            y=0.0,
            button="None",
            state="Move",
            raw_row={},
        )
    start_ts = 10 * 10.0 + gap_ms
    for idx in range(10):
        yield NormalisedEvent(
            dataset_id="synthetic",
            user_id="userA",
            session_id="session1",
            split="train",
            timestamp_ms=start_ts + idx * 10.0,
            x=float(10 + idx) * step,
            y=0.0,
            button="None",
            state="Move",
            raw_row={},
        )


def test_segment_event_stream_gap_split():
    events = list(synthetic_events(gap_ms=500.0))
    gestures = list(
        segment_event_stream(
            events,
            gap_threshold_ms=200.0,
            target_len=16,
        )
    )
    assert len(gestures) == 2
    for gesture in gestures:
        assert gesture.sequence.shape == (16, 3)
        assert gesture.original_event_count >= 5
        assert gesture.metadata["start_xy"] is not None


def make_velocity_demo_events() -> list[NormalisedEvent]:
    events: list[NormalisedEvent] = []
    timestamp = 0.0
    x = 0.0

    def add_event(x_pos: float, t_inc: float = 10.0):
        nonlocal timestamp, x
        timestamp += t_inc
        x = x_pos
        events.append(
            NormalisedEvent(
                dataset_id="synthetic",
                user_id="userA",
                session_id="session1",
                split="train",
                timestamp_ms=timestamp,
                x=x,
                y=0.0,
                button="None",
                state="Move",
                raw_row={},
            )
        )

    # initial seed event
    events.append(
        NormalisedEvent(
            dataset_id="synthetic",
            user_id="userA",
            session_id="session1",
            split="train",
            timestamp_ms=0.0,
            x=0.0,
            y=0.0,
            button="None",
            state="Move",
            raw_row={},
        )
    )

    # high velocity segment
    for _ in range(5):
        add_event(x + 5.0)

    # slow movement (should trigger idle accumulation)
    for _ in range(6):
        add_event(x + 0.1)

    # high velocity again
    for _ in range(5):
        add_event(x + 5.0)

    return events


def test_segment_event_stream_velocity_threshold():
    events = make_velocity_demo_events()
    gestures = list(
        segment_event_stream(
            events,
            gap_threshold_ms=500.0,
            target_len=16,
            velocity_start_threshold=300.0,
            velocity_stop_threshold=50.0,
            velocity_idle_time_ms=30.0,
        )
    )
    assert len(gestures) >= 2


def test_events_to_gesture_padding_mode():
    events = list(islice(synthetic_events(gap_ms=0.0), 6))
    gesture = events_to_gesture(
        events,
        target_len=16,
        resample_strategy="pad",
        padding_mode="zero",
    )
    assert gesture is not None
    assert gesture.sequence.shape == (16, 3)
    assert np.all(gesture.mask[:5] == 1.0)
    assert np.all(gesture.mask[5:] == 0.0)

