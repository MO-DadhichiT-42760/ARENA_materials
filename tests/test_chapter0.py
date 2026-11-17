"""Run the curriculum's own tests against reference solutions, without downloads."""

import importlib
import inspect
from pathlib import Path

import pytest
import torch as t

SECTIONS = [
    "part0_prereqs",
    "part1_ray_tracing",
    "part2_cnns",
    "part3_optimization",
    "part4_backprop",
    "part51_vaes",
    "part52_gans",
]


def curriculum_cases():
    cases = []
    excluded = {
        "test_broadcast",
        "test_reduce",
        "test_all_reduce",
        "test_step_doesnt_modify_grads",
        "test_initialize_weights",
    }
    for section in SECTIONS:
        solutions = importlib.import_module(f"{section}.solutions")
        tests = importlib.import_module(f"{section}.tests")
        for name, function in inspect.getmembers(tests, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            arguments = {}
            for parameter in inspect.signature(function).parameters.values():
                if (
                    parameter.default is not inspect.Parameter.empty
                    or parameter.kind == parameter.VAR_KEYWORD
                ):
                    continue
                if hasattr(solutions, parameter.name):
                    arguments[parameter.name] = getattr(solutions, parameter.name)
                else:
                    assert name in excluded, (
                        f"Uncovered curriculum test: {section}.{name} ({parameter.name})"
                    )
                    break
            else:
                cases.append(pytest.param(function, arguments, id=f"{section}.{name}"))
    return cases


@pytest.mark.parametrize("function, arguments", curriculum_cases())
def test_curriculum(function, arguments):
    t.set_num_threads(4)
    function(**arguments)


def test_local_assets():
    from scripts.chapter0_assets import check

    check()


def test_native_forward_backward():
    """Exercise actual MPS/CPU training operations, including VAEs and GANs."""
    from part2_cnns.solutions import SimpleMLP
    from part51_vaes.solutions import VAE
    from part52_gans.solutions import DCGAN

    device = t.device("mps" if t.backends.mps.is_available() else "cpu")
    models = [
        (SimpleMLP(), t.randn(4, 1, 28, 28)),
        (VAE(latent_dim_size=4, hidden_dim_size=16), t.randn(4, 1, 28, 28)),
    ]
    for model, images in models:
        model = model.to(device)
        optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
        result = model(images.to(device))
        output = result[0] if isinstance(result, tuple) else result
        loss = output.square().mean()
        assert t.isfinite(loss).item()
        loss.backward()
        optimizer.step()

    gan = DCGAN(latent_dim_size=8, img_size=28, img_channels=1, hidden_channels=[8, 16]).to(device)
    optimizer = t.optim.Adam(gan.parameters(), lr=1e-3)
    generated = gan.netG(t.randn(4, 8, device=device))
    logits = gan.netD(generated)
    loss = logits.square().mean()
    assert t.isfinite(loss).item()
    loss.backward()
    optimizer.step()


def test_portable_data():
    from fundamentals_utils import CIFAR10, MNIST

    for dataset_class in (MNIST, CIFAR10):
        data = dataset_class(train=False)
        batch = next(iter(t.utils.data.DataLoader(data, batch_size=2)))
        assert batch[0].dtype == t.uint8
        assert data.transform(batch[0]).dtype == t.float32
    from part4_backprop.utils import get_mnist

    train, _test = get_mnist(subsample=100)
    assert next(iter(train))[0].shape[1:] == (1, 28, 28)


@pytest.mark.parametrize(
    "name, config",
    [
        ("SGD", {"momentum": 0.9}),
        ("RMSprop", {"momentum": 0.9}),
        ("Adam", {}),
        ("AdamW", {}),
    ],
)
def test_optimizer_keeps_gradients(name, config):
    from part3_optimization import solutions, tests

    tests.test_step_doesnt_modify_grads(getattr(solutions, name), **config)


def test_gan_weight_initialization():
    from part52_gans.solutions import initialize_weights
    from part52_gans.tests import test_initialize_weights

    test_initialize_weights(
        initialize_weights, t.nn.ConvTranspose2d, t.nn.Conv2d, t.nn.Linear, t.nn.BatchNorm2d
    )


def test_notebook_setup_and_syntax():
    import ast
    import json

    import nbformat

    root = Path(__file__).resolve().parents[1]
    setups = set()
    notebooks = list((root / "chapter0_fundamentals/exercises").rglob("*.ipynb"))
    assert len(notebooks) == 14
    for path in notebooks:
        notebook = json.loads(path.read_text())
        nbformat.validate(notebook)
        assert notebook["metadata"]["kernelspec"]["name"] == "arena-chapter0"
        setup = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "".join(cell["source"]).startswith("# Native local setup:")
        ]
        assert len(setup) == 1, path
        setups.add(ast.dump(ast.parse(setup[0])))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))
    assert len(setups) == 1, "Notebook setup cells differ; regenerate from the master files"
