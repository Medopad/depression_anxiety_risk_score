"""This is a test module for the main module of the package."""
from typing import Callable

from _pytest.capture import CaptureResult
import pytest

from depression_anxiety_risk_scores.main import hello_world, start


@pytest.fixture
def expected_text() -> str:
    """Returns expected text to be printed out.

    This function returns the expected text to check
    for the value of the hello_world.

    Returns:
        str: The expected text to be printed.

    """
    return "Hello, World!"


def test_if_hello_world_is_typed_right(expected_text: Callable[[], str]) -> None:
    """Checks mispellings of hello_world.

    This function checks whether the value of the hello_world variable
    is typed right.

    Args:
        expected_text(Callable[[], str]): The pytest fixture
            which returns the expexted text
    """
    assert hello_world == expected_text


def test_if_hello_world_is_printed_to_the_screen(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Checks printed value.

    This function checks whether the value of the hello_world variable is printed
    to the standard output.

    Args:
        capfd(pytest.CaptureFixture[str]): The capture fixture
            which helps to read the stdout
    """
    start()
    std_result: CaptureResult = capfd.readouterr()
    assert std_result.out.rstrip() == hello_world
