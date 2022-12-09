"""This is the nox setup for the  package."""
from tempfile import NamedTemporaryFile
from typing import Any, IO

import nox
from nox.sessions import Session


package = "depression_anxiety_risk_scores"
locations = "src", "tests", "noxfile.py", "docs/conf.py"
nox.options.sessions = (
    "check_package_safety",
    "check_types",
    "run_tests",
    "check_code_formatting",
)


@nox.session(python="3.10")
def check_package_safety(session: Session) -> None:
    """Scan dependencies for insecure packages.

    Args:
        session(Session): The Session object.
    """
    requirements: IO[bytes]
    with NamedTemporaryFile() as requirements:
        install_with_constraints_using_requirements_file(
            requirements, session, "safety"
        )
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python="3.10")
def check_types(session: Session) -> None:
    """Type-check using mypy and typeguard.

    Args:
        session(Session): The Session object.
    """
    mypy_args = session.posargs or locations
    typeguard_args = session.posargs
    poetry_install(session)
    install_with_constraints(session, "mypy")
    install_with_constraints(session, "pytest", "typeguard")
    session.run("mypy", *mypy_args)
    session.run("pytest", f"--typeguard-packages={package}", *typeguard_args)


@nox.session(python="3.10")
def run_tests(session: Session) -> None:
    """Run the unit tests and the document tests.

    Args:
        session(Session): The Session object.
    """
    unit_test_args = session.posargs or ["--cov"]
    xdoctest_args = session.posargs or ["all"]
    poetry_install(session)
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
        "xdoctest",
        "pygments",
        "colorama",
    )
    session.run("pytest", *unit_test_args)
    session.run("python", "-m", "xdoctest", package, *xdoctest_args)


@nox.session(python="3.10")
def check_code_formatting(session: Session) -> None:
    """Lint using flake8.

    Args:
        session(Session): The Session object.
    """
    args = session.posargs or locations
    install_with_constraints(
        session,
        "darglint",
        "dlint",
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-cognitive-complexity",
        "flake8-docstrings",
        "flake8-expression-complexity",
        "flake8-functions",
        "flake8-import-order",
        "flake8-simplify",
        "flake8-spellcheck",
    )
    session.run("flake8", *args)


@nox.session(python="3.10")
def create_documentation(session: Session) -> None:
    """Build the documentation.

    Args:
        session(Session): The Session object.
    """
    from os import path
    from shutil import rmtree

    build_path = path.join(path.dirname(path.abspath(__file__)), "docs/_build/")
    summary_path = path.join(path.dirname(path.abspath(__file__)), "docs/_autosummary/")
    if path.exists(build_path):
        rmtree(build_path)
    if path.exists(summary_path):
        rmtree(summary_path)

    poetry_install(session)
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")


def install_with_constraints(
    session: Session, *args: str, **kwargs: Any  # noqa: ANN401
) -> None:
    """Run install_with_constraints_using_requirements_file with a requirements file.

    This function creates a temporary requirements file
    to call install_with_constraints_using_requirements_file with it.

    Args:
        session(Session): The Session object.
        args(str): Command-line arguments for pip.
        kwargs(Any): Additional keyword arguments for Session.install.
    """
    requirements: IO[bytes]
    with NamedTemporaryFile() as requirements:
        install_with_constraints_using_requirements_file(
            requirements, session, *args, **kwargs
        )


def install_with_constraints_using_requirements_file(
    requirements: IO[bytes], session: Session, *args: str, **kwargs: Any  # noqa: ANN401
) -> None:
    """Install packages constrained by Poetry's lock file.

    Args:
        requirements(IO[bytes]): The existing requirements file.
        session(Session): The Session object.
        args(str): Command-line arguments for pip.
        kwargs(Any): Additional keyword arguments for Session.install.

    """
    export_requirements_txt(session, requirements)
    session.install("-r", requirements.name, *args, **kwargs)


def export_requirements_txt(session: Session, requirements: IO[bytes]) -> None:
    """Export the poetry packages as the requirements.txt file so that nox can use it.

    Args:
        session(Session): The Session object.
        requirements(IO[bytes]): The existing requirements file.

    """
    session.run(
        "poetry",
        "export",
        "--without-hashes",
        "--with",
        "dev",
        "--format=requirements.txt",
        f"--output={requirements.name}",
        external=True,
    )


def poetry_install(session: Session) -> None:
    """Installs poetry in session.

    Args:
        session(Session): The Session object.

    """
    session.run("poetry", "install", "--no-dev", external=True)
