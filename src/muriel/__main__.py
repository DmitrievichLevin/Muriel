"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Muriel."""


if __name__ == "__main__":
    raise Exception()
    main(prog_name="Burgos_ECS")  # pragma: no cover
