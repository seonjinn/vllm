# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import prometheus_client
import pytest

import vllm.envs as envs
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.reader import (
    Counter,
    Gauge,
    Histogram,
    Vector,
    get_metrics_snapshot,
)
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def test_registry(monkeypatch):
    # Use a custom registry for tests
    test_registry = prometheus_client.CollectorRegistry(auto_describe=True)
    monkeypatch.setattr("vllm.v1.metrics.reader.REGISTRY", test_registry)
    return test_registry


def _make_prometheus_logger(test_registry, monkeypatch):
    for metric_cls_name in ("_counter_cls", "_gauge_cls", "_histogram_cls"):
        metric_cls = getattr(PrometheusStatLogger, metric_cls_name)
        monkeypatch.setattr(
            PrometheusStatLogger,
            metric_cls_name,
            partial(metric_cls, registry=test_registry),
        )

    config = SimpleNamespace(
        cache_config=SimpleNamespace(),
        kv_transfer_config=None,
        lora_config=None,
        model_config=SimpleNamespace(
            is_diffusion=False,
            max_model_len=128,
            served_model_name="test-model",
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=False,
            show_hidden_metrics=False,
        ),
        speculative_config=None,
    )
    return PrometheusStatLogger(config, engine_indexes=[0])


def test_dynamic_sd_scheduler_key_counter(test_registry, monkeypatch):
    monkeypatch.setenv("VLLM_DYNAMIC_SD_PROFILE_METRICS", "1")
    logger = _make_prometheus_logger(test_registry, monkeypatch)

    logger.record(
        SchedulerStats(
            num_scheduled_reqs=4,
            num_spec_tokens_to_schedule=3,
        ),
        iteration_stats=None,
    )

    metric = next(
        metric
        for metric in get_metrics_snapshot()
        if metric.name == "vllm:dynamic_sd_scheduler_steps"
    )
    assert isinstance(metric, Counter)
    assert metric.value == 1
    assert metric.labels["engine"] == "0"
    assert metric.labels["scheduler_batch_size"] == "4"
    assert metric.labels["k"] == "3"


def test_dynamic_sd_scheduler_key_counter_defaults_off(test_registry, monkeypatch):
    monkeypatch.delenv("VLLM_DYNAMIC_SD_PROFILE_METRICS", raising=False)
    assert not envs.VLLM_DYNAMIC_SD_PROFILE_METRICS
    logger = _make_prometheus_logger(test_registry, monkeypatch)

    logger.record(
        SchedulerStats(
            num_scheduled_reqs=4,
            num_spec_tokens_to_schedule=3,
        ),
        iteration_stats=None,
    )

    assert all(
        metric.name != "vllm:dynamic_sd_scheduler_steps"
        for metric in get_metrics_snapshot()
    )


@pytest.mark.parametrize("num_engines", [1, 4])
def test_gauge_metric(test_registry, num_engines):
    g = prometheus_client.Gauge(
        "vllm:test_gauge",
        "Test gauge metric",
        labelnames=["model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        g.labels(model="foo", engine_index=str(i)).set(98.5)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Gauge)
        assert m.name == "vllm:test_gauge"
        assert m.value == 98.5
        assert m.labels["model"] == "foo"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_counter_metric(test_registry, num_engines):
    c = prometheus_client.Counter(
        "vllm:test_counter",
        "Test counter metric",
        labelnames=["model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        c.labels(model="bar", engine_index=str(i)).inc(19)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Counter)
        assert m.name == "vllm:test_counter"
        assert m.value == 19
        assert m.labels["model"] == "bar"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_histogram_metric(test_registry, num_engines):
    h = prometheus_client.Histogram(
        "vllm:test_histogram",
        "Test histogram metric",
        labelnames=["model", "engine_index"],
        buckets=[10, 20, 30, 40, 50],
        registry=test_registry,
    )
    for i in range(num_engines):
        hist = h.labels(model="blaa", engine_index=str(i))
        hist.observe(42)
        hist.observe(21)
        hist.observe(7)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Histogram)
        assert m.name == "vllm:test_histogram"
        assert m.count == 3
        assert m.sum == 70
        assert m.buckets["10.0"] == 1
        assert m.buckets["20.0"] == 1
        assert m.buckets["30.0"] == 2
        assert m.buckets["40.0"] == 2
        assert m.buckets["50.0"] == 3
        assert m.labels["model"] == "blaa"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])


@pytest.mark.parametrize("num_engines", [1, 4])
def test_vector_metric(test_registry, num_engines):
    c = prometheus_client.Counter(
        "vllm:spec_decode_num_accepted_tokens_per_pos",
        "Vector-like counter metric",
        labelnames=["position", "model", "engine_index"],
        registry=test_registry,
    )
    for i in range(num_engines):
        c.labels(position="0", model="llama", engine_index=str(i)).inc(10)
        c.labels(position="1", model="llama", engine_index=str(i)).inc(5)
        c.labels(position="2", model="llama", engine_index=str(i)).inc(1)

    metrics = get_metrics_snapshot()
    assert len(metrics) == num_engines
    engine_labels = [str(i) for i in range(num_engines)]
    for m in metrics:
        assert isinstance(m, Vector)
        assert m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
        assert m.values == [10, 5, 1]
        assert m.labels["model"] == "llama"
        assert m.labels["engine_index"] in engine_labels
        engine_labels.remove(m.labels["engine_index"])
