"""
aiplasma.cli.commands.init
`aiplasma init <project-name>` (ADR-013).

Scaffolds a full project structure for someone building on top of
AIPlasma as an installed library, in the current working directory.
"""

from pathlib import Path

import click

from .. import templates


@click.command()
@click.argument("project_name")
def init(project_name: str) -> None:
    """Scaffold a new project that depends on AIPlasma."""
    root = Path.cwd() / project_name
    if root.exists():
        raise click.ClickException(f"{root} already exists — refusing to overwrite it.")

    root.mkdir(parents=True)
    (root / "checkpoints").mkdir()

    (root / "pyproject.toml").write_text(
        templates.init_pyproject_template(project_name), encoding="utf-8"
    )
    (root / "main.py").write_text(
        templates.init_main_template(project_name), encoding="utf-8"
    )
    (root / "README.md").write_text(
        templates.init_readme_template(project_name), encoding="utf-8"
    )

    click.echo(f"Project '{project_name}' created in {root}/")
    click.echo(f"\n  cd {project_name}")
    click.echo("  pip install -e .")
    click.echo("  python main.py")
