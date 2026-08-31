#!/usr/bin/env python3
"""Print reproducible per-layer trainable parameter counts for every entry point.

The report counts every parameter owned by QuantGate and its domain discriminator,
including biases and normalization affine parameters. It prints registered and
trainable totals, and does not estimate a count from paper notation or omit the
discriminator.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "BCI_14_7513_sota": {
        "emb_size": 120,
        "depth": 6,
        "window_size": 5,
        "n_classes": 2,
        "domain_classes": 13,
        "expected": {"QuantGate": 9501982, "Discriminator": 4557, "combined": 9506539},
    },
    "BCI_2a9_7922_sota": {
        "emb_size": 253,
        "depth": 2,
        "window_size": 5,
        "n_classes": 2,
        "domain_classes": 8,
        "expected": {"QuantGate": 4006740, "Discriminator": 2344, "combined": 4009084},
    },
    "BCI_IVA5_8479_sota": {
        "emb_size": 7021,
        "depth": 6,
        "window_size": 1,
        "n_classes": 2,
        "domain_classes": 4,
        "expected": {"QuantGate": 55313328, "Discriminator": 4260, "combined": 55317588},
    },
    "BCI_IVNO7_8293_sota": {
        "emb_size": 1770,
        "depth": 6,
        "window_size": 1,
        "n_classes": 2,
        "domain_classes": 6,
        "expected": {"QuantGate": 20268154, "Discriminator": 4326, "combined": 20272480},
    },
}


def load_entry_point(name):
    """Load one main.py without executing its training entry point."""
    entry_dir = ROOT / name
    local_roots = {"modules", "dataloader", "Adver_network", "utils", "model", "core_qnn"}
    for loaded_name in list(sys.modules):
        if loaded_name.split(".", 1)[0] in local_roots:
            del sys.modules[loaded_name]
    sys.path.insert(0, str(entry_dir))
    module_name = "q_mixer_" + name.lower()
    spec = importlib.util.spec_from_file_location(module_name, entry_dir / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {entry_dir / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    # QuantGate reads this legacy module-global during construction.
    module.args = argparse.Namespace(window_size=CONFIGS[name]["window_size"])
    spec.loader.exec_module(module)
    return module


def direct_parameter_rows(model):
    """Return parameter counts owned directly by each module (no double count)."""
    rows = []
    for name, submodule in model.named_modules():
        count = sum(
            parameter.numel()
            for parameter in submodule.parameters(recurse=False)
            if parameter.requires_grad
        )
        if count:
            rows.append((name or "<root>", count))
    return rows


def count(model, trainable_only=True):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def report(name):
    config = CONFIGS[name]
    module = load_entry_point(name)
    model = module.QuantGate(
        emb_size=config["emb_size"],
        depth=config["depth"],
        bottleneck_dim=128,
        n_classes=config["n_classes"],
    )
    discriminator = module.Discriminator(
        emb_size=config["emb_size"],
        n_classes=config["domain_classes"],
    )

    print(f"\n{name}")
    print("config:", config, "D_q=512")
    for label, network in (("QuantGate", model), ("Discriminator", discriminator)):
        print(f"{label} layers:")
        for layer, parameters in direct_parameter_rows(network):
            print(f"  {layer:55s} {parameters:>12,d}")
        print(f"{label} registered total: {count(network, trainable_only=False):,}")
        print(f"{label} trainable total: {count(network):,}")
    totals = {
        "QuantGate": count(model),
        "Discriminator": count(discriminator),
    }
    totals["combined"] = totals["QuantGate"] + totals["Discriminator"]
    print(f"Combined trainable total: {totals['combined']:,}")
    if totals != config["expected"]:
        raise RuntimeError(f"Parameter count changed for {name}: {totals} != {config['expected']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry-point",
        choices=sorted(CONFIGS),
        help="Report only one dataset entry point (default: all four)",
    )
    args = parser.parse_args()
    names = [args.entry_point] if args.entry_point else sorted(CONFIGS)
    for name in names:
        report(name)


if __name__ == "__main__":
    main()
