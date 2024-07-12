![SVS Logo](https://raw.githubusercontent.com/Rhobota/svs/main/logos/svs.png)

# Stupid Vector Store (SVS)

[![PyPI - Version](https://img.shields.io/pypi/v/svs.svg)](https://pypi.org/project/svs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/svs.svg)](https://pypi.org/project/svs)
![Test Status](https://github.com/Rhobota/svs/actions/workflows/test.yml/badge.svg?branch=main)
[![Downloads](https://static.pepy.tech/badge/svs)](https://pepy.tech/project/svs)

- 💩 Why is it _stupid_?
  - Because it just uses [SQLite](https://www.sqlite.org/) and [NumPy](https://numpy.org/). Nothing fancy.
  - That is our core design choice. We want something _stupid simple_, yet _reasonably fast_.

- 🧠 Is it possibly... _smart_ in any way though?
  - Maybe.
  - **It will squeeze the most juice from your machine: 🍊**
     - Optimized SQL
     - Cache-friendly memory access
     - Fast in the places that matter 🚀
     - All with a simple Python interface

## Overview

SVS is stupid yet can handle a million documents on commodity hardware, so it's probably perfect for you.

**Should you use SVS?** SVS is designed for the use-case where:
 1. you have less than a million documents, and
 2. you don't add/remove documents very often.

If that's you, then SVS will probably be the simples (and _stupidest_) way to manage your document vectors!

## Table of Contents

- [Installation](#installation)
- [Used By](#used-by)
- [Quickstart](#quickstart)
- [Debug Logging](#debug-logging)
- [License](#license)

## Installation

```console
pip install -U svs
```

## Used By

SVS is used in production by:

[![AutoAuto](https://raw.githubusercontent.com/Rhobota/svs/main/logos/autoauto.png)](https://www.autoauto.ai/)

## Quickstart

TODO

## Debug Logging

This library logs using Python's builtin `logging` module. It logs mostly to `INFO`, so here's a snippet of code you can put in _your_ app to see those traces:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# ... now use SVS as you normally would, but you'll see extra log traces!
```

## License

`svs` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
