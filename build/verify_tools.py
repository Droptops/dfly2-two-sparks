#!/usr/bin/env python3
"""Check the scoped request-tool dependencies without auditing unrelated NGC extras."""
import importlib.metadata as md
import json
from pathlib import Path
from urllib.parse import urlparse

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename


def require(name, spec=""):
    actual = md.version(name)
    if spec and actual not in SpecifierSet(spec):
        raise RuntimeError(f"{name}=={actual} does not satisfy {spec}")
    return actual


def main():
    checked = {}
    for line in Path(__file__).with_name("tool-requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        url = line.split(" @ ", 1)[1].split()[0]
        name, version, _, _ = parse_wheel_filename(Path(urlparse(url).path).name)
        checked[name] = require(name, "==" + str(version))
        for text in md.requires(name) or []:
            dep = Requirement(text)
            if dep.marker is not None and not dep.marker.evaluate({"extra": ""}):
                continue
            checked[dep.name] = require(dep.name, str(dep.specifier))
    checked["jinja2"] = require("jinja2", ">=3.1.0")
    checked["filelock"] = require("filelock")
    try:
        checked["accelerate"] = require("accelerate", ">=1.1.0")
    except md.PackageNotFoundError:
        pass
    from transformers import AutoTokenizer  # noqa: F401 -- actual lazy-import path
    import jinja2  # noqa: F401
    import tokenizers  # noqa: F401
    print(json.dumps({"request_tool_dependencies": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
