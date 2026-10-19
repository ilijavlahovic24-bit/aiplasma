"""
aiplasma.cli.commands.new
`aiplasma new problem|model|trainer|example <Name> [options]` (ADR-013).

Always writes to the current working directory.
"""

from pathlib import Path

import click

from .. import templates


def _registered_equations() -> list[str]:
    """Equation names from REGISTRY, or an empty list if AIPlasma isn't importable here."""
    try:
        from solvers.physics_constraints import REGISTRY
        return REGISTRY.list_all()
    except ImportError:
        return []


def _write(path: Path, content: str) -> None:
    if path.exists():
        raise click.ClickException(f"{path} already exists — refusing to overwrite it.")
    path.write_text(content, encoding="utf-8")
    click.echo(f"created {path}")


@click.group()
def new() -> None:
    """Scaffold a new problem, model, trainer, or full example."""


@new.command("problem")
@click.argument("name")
@click.option("--equation", required=True, help="Registered PDE equation name (see `aiplasma list equations`).")
def new_problem(name: str, equation: str) -> None:
    """Generate a PhysicsProblem subclass wired to --equation."""
    known = _registered_equations()
    if known and equation not in known:
        raise click.ClickException(
            f"'{equation}' is not a registered equation. Known equations: {', '.join(known)}"
        )
    content = templates.problem_template(name, equation)
    _write(Path.cwd() / f"{_snake(name)}.py", content)


@new.command("model")
@click.argument("name")
@click.option(
    "--type", "model_type", required=True,
    type=click.Choice(sorted(templates.MODEL_BASES)),
    help="Base class this model inherits from.",
)
def new_model(name: str, model_type: str) -> None:
    """Generate a PhysicsModel subclass of the chosen --type."""
    content = templates.model_template(name, model_type)
    _write(Path.cwd() / f"{_snake(name)}.py", content)


@new.command("trainer")
@click.argument("name")
@click.option(
    "--type", "trainer_type", required=True,
    type=click.Choice(sorted(templates.TRAINER_BASES)),
    help="Base trainer this subclass extends.",
)
def new_trainer(name: str, trainer_type: str) -> None:
    """Generate a Trainer subclass of the chosen --type."""
    content = templates.trainer_template(name, trainer_type)
    _write(Path.cwd() / f"{_snake(name)}.py", content)


@new.command("example")
@click.argument("name")
@click.option("--equation", required=True, help="Registered PDE equation name (see `aiplasma list equations`).")
def new_example(name: str, equation: str) -> None:
    """Generate a full example folder: README, train/evaluate/visualize scripts, config.yaml."""
    known = _registered_equations()
    if known and equation not in known:
        raise click.ClickException(
            f"'{equation}' is not a registered equation. Known equations: {', '.join(known)}"
        )

    folder = Path.cwd() / _snake(name)
    if folder.exists():
        raise click.ClickException(f"{folder} already exists — refusing to overwrite it.")
    folder.mkdir(parents=True)

    _write(folder / "problem.py", templates.problem_template(f"{name}Problem", equation))
    _write(folder / "config.yaml", templates.example_config_template(equation))
    _write(folder / "train.py", templates.example_train_template(name, equation))
    _write(folder / "evaluate.py", templates.example_evaluate_template(name))
    _write(folder / "visualize.py", templates.example_visualize_template(name))
    _write(folder / "README.md", templates.example_readme_template(name, equation))

    click.echo(f"\nExample '{name}' ready in {folder}/")


def _snake(name: str) -> str:
    """CamelCase -> snake_case, for file names."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
