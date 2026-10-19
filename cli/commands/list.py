"""
aiplasma.cli.commands.list_cmd
`aiplasma list equations|models` (ADR-013).
"""

import click

from .. import templates


@click.group()
def list_() -> None:
    """List registered equations or available model types."""


@list_.command("equations")
def list_equations() -> None:
    """List all equations registered in the PDE registry."""
    try:
        from solvers.physics_constraints import REGISTRY
    except ImportError as exc:
        raise click.ClickException(
            "Could not import AIPlasma from this directory. "
            "Run this command from inside an AIPlasma project."
        ) from exc

    for name in REGISTRY.list_all():
        eq = REGISTRY.get(name)
        click.echo(f"{name:22s} params={eq.expected_params()}")
        click.echo(f"{'':22s} {eq.description()}")


@list_.command("models")
def list_models() -> None:
    """List the model base classes available to `aiplasma new model --type`."""
    for type_name, (class_name, module) in sorted(templates.MODEL_BASES.items()):
        click.echo(f"{type_name:16s} -> {class_name}  ({module})")
