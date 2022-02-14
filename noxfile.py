"""Nox automation file."""

from nox import Session, session

python_versions = ["3.10", "3.9", "3.8", "3.7"]


@session()
def format_check(session: Session) -> None:
    """Check for black format compliance."""
    session.install("black")
    session.run("black", "bfio", "--check")


@session(python=["3.9"])
def lint_check(session: Session) -> None:
    """Lint checks"""
    session.install("flake8")
    session.run(
        "flake8",
        "bfio",
        "--count",
        "--ignore=F722",
        "--select=E9,F63,F7,F82",
        "--show-source",
        "--statistics",
    )
    session.run(
        "flake8",
        "bfio",
        "--count",
        "--exit-zero",
        "--max-line-length=127",
        "--statistics",
    )
