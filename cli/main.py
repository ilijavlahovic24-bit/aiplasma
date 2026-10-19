"""
aiplasma.cli.main
Root click group for the `aiplasma` command (ADR-013).

    aiplasma init
    aiplasma new <problem|model|trainer|example> <Name> [options]
    aiplasma list <equations|models>
"""

import click

from .commands.init import init
from .commands.new import new
from .commands.list_cmd import list_


@click.group()
@click.version_option(package_name="aiplasma")
def main() -> None:
    """AIPlasma — physics-informed ML framework for plasma physics and fusion."""


main.add_command(init)
main.add_command(new)
main.add_command(list_, name="list")


if __name__ == "__main__":
    main()
