"""Test log feedback (GĐ1): logger JSONL/Null + analytics CTR."""

from __future__ import annotations

from news_search.config import Settings
from news_search.feedback import get_event_logger
from news_search.feedback.analytics import compute_metrics
from news_search.feedback.base import ClickEvent, SearchEvent
from news_search.feedback.logger import JsonlEventLogger, NullEventLogger


def test_disabled_returns_null():
    assert isinstance(get_event_logger(Settings(feedback_enabled=False)), NullEventLogger)


def test_factory_jsonl(tmp_path):
    s = Settings(feedback_enabled=True, feedback_backend="jsonl",
                 feedback_path=str(tmp_path / "e.jsonl"))
    assert isinstance(get_event_logger(s), JsonlEventLogger)


def test_jsonl_and_analytics(tmp_path):
    path = str(tmp_path / "events.jsonl")
    lg = JsonlEventLogger(path)
    lg.log_search(SearchEvent("s1", "lạm phát", "hybrid", 10, ["a", "b"], 2, False))
    lg.log_search(SearchEvent("s2", "xyz", "hybrid", 10, [], 0, True))  # zero-result
    lg.log_click(ClickEvent("s1", "a", 1, dwell_ms=3000))

    m = compute_metrics(path)
    assert m["searches"] == 2
    assert m["clicks"] == 1
    assert m["zero_result_rate"] == 0.5
    assert m["ctr"] == 0.5  # 1/2 lượt tìm có click
    assert m["clicks_by_position"] == {1: 1}
    assert m["avg_dwell_ms"] == 3000.0


def test_analytics_missing_file():
    assert compute_metrics("/khong/ton/tai.jsonl")["searches"] == 0
