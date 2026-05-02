"""Module entry: ``python -m qed_nlce`` dispatches to the unified CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
