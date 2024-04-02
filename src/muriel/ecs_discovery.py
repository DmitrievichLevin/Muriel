import os
import sys
from fnmatch import fnmatch
from inspect import getmembers, isclass
import importlib.util
import logging


def norecursedirs(x: str) -> bool:
    """Ignore Directories and/or Files on Walk

    Args:
        x (str): dir/file

    Returns:
        (bool): whether dir/file is valid
    """
    is_dir = os.path.isdir(x)
    ignore = not any(
        fnmatch(x, glob)
        for glob in [
            "*.egg",
            "*.md",
            "_darcs",
            "build",
            "CVS",
            "dist",
            "*cache*",
            "tests",
            "node_modules",
            "venv",
            "{arch}",
            ".*",
            "*.typed",
            "docs",
            "coverage*",
            "__init__",
            "ecs_**.py",
        ]
    )

    exceptions = any(
        fnmatch(x, glob) for glob in [".py", "ecs_builtin_states.py"]
    )
    if exceptions:
        return True
    return is_dir and ignore


def get_subclasses(cls: type, root, *files) -> list[tuple[str, type]]:
    """Get all subclasses in directory

    - lazy importing docs
    - https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    Args:
        root (str): dir of files

    Returns:
        list[tuple[str, type]]: (name, cls), ...
    """
    sub_classes = []
    for mod in files:
        if norecursedirs(mod):
            spec = importlib.util.spec_from_file_location(
                mod, "".join([root, "/", mod])
            )
            if spec:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]

            members = getmembers(
                module,
                lambda x: isclass(x)
                and issubclass(x, cls)
                and x is not cls,
            )
            sub_classes += members
    return sub_classes


def submodule_walk(cls: type) -> list[tuple[str, type]]:
    """Get all subclasses in current working directory

    Args:
        cls (type): super class

    Returns:
        list[tuple[str, type]]: _description_
    """
    cwd = os.getcwd()

    modules = []
    for root, dirs, files in os.walk(cwd + "/src"):
        if not norecursedirs(root):
            continue

        subs = get_subclasses(cls, root, *files)
        modules += subs
        logging.debug(
            "Sub-module Walking... \nPath: %s \nDirectories: %s \nFound Modules: %s \n",
            root,
            dirs,
            list(filter(norecursedirs, files)),
        )

    logging.debug(
        "type[%s]\nModules found: %s", cls.__class__.__name__, modules
    )

    return modules
