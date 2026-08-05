from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.model_loader import dummy_loader


def test_dummy_loader_processes_serialized_routed_experts(monkeypatch):
    class FakeRoutedExperts:
        def __init__(self):
            self.quant_method = Mock()

    layer = FakeRoutedExperts()
    model = SimpleNamespace(modules=lambda: [layer])
    model_config = Mock()

    monkeypatch.setattr(dummy_loader, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(
        dummy_loader,
        "get_layerwise_info",
        lambda _: SimpleNamespace(can_load=lambda: False),
    )
    initialize_dummy_weights = Mock()
    monkeypatch.setattr(
        dummy_loader, "initialize_dummy_weights", initialize_dummy_weights
    )

    loader = object.__new__(dummy_loader.DummyModelLoader)
    loader.load_weights(model, model_config)

    initialize_dummy_weights.assert_called_once_with(layer, model_config)
    layer.quant_method.process_weights_after_loading.assert_called_once_with(layer)
