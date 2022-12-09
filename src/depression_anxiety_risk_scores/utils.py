import os
from jinja2 import Environment, BaseLoader


def get_absolute_path(rel_path: str) -> str:
    """Convert relative path to absolute path.

    Args:
        rel_path (str): relative path

    Returns:
        (str): absolute path.
    """
    cur_dir = os.path.dirname(__file__)
    return os.path.join(cur_dir, rel_path)


def read_relative_path(path: str) -> str:
    """Read a file from a relative path.

    Args:
        path (str): relative file path

    Returns:
        (str): loaded contents.
    """
    with open(get_absolute_path(path)) as file:
        return file.read()


def read_and_render(path: str, **kwargs) -> str:
    """Read and render a jinja template.

    Args:
        path (str): relative path to jinja template.

    Returns:
     (str): rendered template string.
    """
    template = Environment(loader=BaseLoader).from_string(read_relative_path(path))
    return template.render(**kwargs)
