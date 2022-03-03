"""Nox automation file."""

from nox import Session, session
from pathlib import Path
import shutil

python_versions = ["3.10", "3.9", "3.8", "3.7"]


@session(python=["3.9"])
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


@session(python=["3.9"])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["html"]
    session.install("-e", ".")
    session.install("-r", "requirements/requirements-docs.txt")
    session.install("sphinx")

    source_dir = Path("docs", "source")
    build_dir = Path("docs", "build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", "-b", "html", str(source_dir), str(build_dir))
