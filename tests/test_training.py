"""Smoke and regression tests for the released MI training entry points."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("BCI_14_7513_sota", 14, 5, 120),
    ("BCI_2a9_7922_sota", 9, 5, 253),
    ("BCI_IVA5_8479_sota", 5, 1, 7021),
    ("BCI_IVNO7_8293_sota", 7, 1, 1770),
]
LOCAL_MODULES = {
    "Adver_network", "Weight", "centroid_align", "core_qnn", "dataloader",
    "datasets_", "modules", "utils",
}


@pytest.fixture(params=CONFIGS, ids=[row[0] for row in CONFIGS])
def entry(request, monkeypatch):
    directory, subjects, windows, features = request.param
    previous = {
        name: module for name, module in list(sys.modules.items())
        if name.split(".")[0] in LOCAL_MODULES
    }
    for name in previous:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(ROOT / directory))
    spec = importlib.util.spec_from_file_location(directory, ROOT / directory / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module, subjects, windows, features
    torch.set_num_threads(old_threads)
    for name in list(sys.modules):
        if name.split(".")[0] in LOCAL_MODULES:
            del sys.modules[name]
    sys.modules.update(previous)


def make_args(subjects, windows):
    return SimpleNamespace(
        target=subjects, window_size=windows, device="cpu", dataset="seed",
        num_class=2, n_epochs=1, batch_size=2, batch_size_fine=2,
        max_iter2=1, seed=1, num_seeds=1,
    )


def test_training_uses_all_sources_and_unpacks_target_batches(entry, monkeypatch):
    module, subjects, windows, features = entry
    args = make_args(subjects, windows)
    monkeypatch.setattr(module, "args", args, raising=False)
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    x = {i: rng.standard_normal((4, windows, features)).astype(np.float32)
         for i in range(subjects)}
    y = {i: np.array([0, 1, 0, 1], dtype=np.int64) for i in range(subjects)}
    batch = {"Tx": torch.from_numpy(x[subjects - 1][:2])}
    for i in range(subjects - 1):
        batch[f"Sx{i + 1}"] = torch.from_numpy(x[i][:2])
        batch[f"Sy{i + 1}"] = torch.from_numpy(y[i][:2])

    class OneBatch:
        num_domains_src = subjects - 1

        def __iter__(self):
            return iter([batch])

    target_test = [{"Tx": torch.from_numpy(x[subjects - 1]),
                    "Ty": torch.from_numpy(y[subjects - 1])}]
    trainer = module.ExGAN(args, args.target, 1)
    monkeypatch.setattr(trainer, "get_source_data",
                        lambda **kwargs: (OneBatch(), target_test, x, y))

    domain_labels = []
    hook = trainer.criterion_domain.register_forward_pre_hook(
        lambda _, inputs: domain_labels.append(inputs[1].detach().clone())
    )
    seen_conditional = []
    conditional = module.utils.conditional

    def capture_conditional(source, target, labels, pseudo_labels, *options):
        seen_conditional.append(source.shape[0])
        return conditional(source, target, labels, pseudo_labels, *options)

    monkeypatch.setattr(module.utils, "conditional", capture_conditional)
    result = trainer.train(1)
    hook.remove()
    assert len(seen_conditional) == subjects - 1
    assert seen_conditional == [args.batch_size] * (subjects - 1)
    assert torch.equal(domain_labels[0], torch.arange(subjects - 1).repeat_interleave(2))
    assert np.isfinite(result[0])
    assert trainer.criterion_domain.num_class == subjects - 1
    assert next(trainer.model.parameters()).device.type == "cpu"

    before = next(trainer.model.parameters()).detach().clone()
    result = trainer.fine_tuning(args, x, y, trainer.model)
    assert np.isfinite(result[:3]).all()
    assert not torch.equal(before, next(trainer.model.parameters()).detach())


def test_smoothed_domain_loss_is_normalized_on_cpu(entry):
    module, subjects, _, _ = entry
    classes = subjects - 1
    logits = torch.randn(4, classes, requires_grad=True)
    labels = torch.arange(4) % classes
    criterion = module.LabelSmooth(num_class=classes)
    actual = criterion(logits, labels)
    expected = F.cross_entropy(logits, labels, label_smoothing=criterion.alpha)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_snapshot_records_target_subject_ids(entry, monkeypatch, tmp_path):
    module, subjects, windows, _ = entry
    args = make_args(subjects, windows)
    recorded_targets = []

    class FakeTrainer:
        def __init__(self, current_args, nsub, fold):
            assert nsub == current_args.target
            recorded_targets.append(current_args.target)

        def train(self, fold):
            return 0.5, 0.5, None, None, {}, {}, None, [0.5]

        def fine_tuning(self, *args):
            return 0.5, 0.5, 0.5, None, None, [0.5], [0.5], [0.5]

    monkeypatch.setattr(module, "ExGAN", FakeTrainer)
    monkeypatch.chdir(tmp_path)
    module.main(args)
    expected = list(range(subjects, 0, -1))
    assert recorded_targets == expected
    lines = (tmp_path / "snapshot.txt").read_text().splitlines()
    assert [line for line in lines if line.startswith("Subject ")] == [
        f"Subject {subject}, seed 1" for subject in expected
    ]


def test_feature_loader_receives_cli_path_and_window(entry, monkeypatch):
    module, subjects, windows, _ = entry
    args = make_args(subjects, windows)
    args.file_path = "custom-feature-directory"
    trainer = module.ExGAN.__new__(module.ExGAN)
    trainer.args = args
    seen = {}

    def fake_load(received_args, **kwargs):
        seen.update(kwargs)
        assert received_args is args
        return 1, 2, 3, 4

    monkeypatch.setattr(module, "load_mi1", fake_load)
    assert trainer.get_source_data() == (1, 2, 3, 4)
    assert seen["path"] == args.file_path
    assert seen["n_windows"] == windows
