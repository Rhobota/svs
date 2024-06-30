![SVS Logo](https://raw.githubusercontent.com/Rhobota/svs/main/logos/svs.png)

# Stupid Vector Store (SVS)

[![PyPI - Version](https://img.shields.io/pypi/v/svs.svg)](https://pypi.org/project/svs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/svs.svg)](https://pypi.org/project/svs)
![Test Status](https://github.com/Rhobota/svs/actions/workflows/test.yml/badge.svg?branch=main)

## Overview

SVS is stupid yet can handle a million documents on commodity hardware, so it's probably perfect for you.

**Should you use SVS?** SVS is designed for the use-case where:
 1. you have less than a million documents, and
 2. you don't add/remove documents very often.

If that's you, then SVS will probably be the simples (and stupidest) way to manage your document vectors!

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
