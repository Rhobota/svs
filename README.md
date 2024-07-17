![SVS Logo](https://raw.githubusercontent.com/Rhobota/svs/main/logos/svs.png)

# Stupid Vector Store (SVS)

[![PyPI - Version](https://img.shields.io/pypi/v/svs.svg)](https://pypi.org/project/svs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/svs.svg)](https://pypi.org/project/svs)
![Test Status](https://github.com/Rhobota/svs/actions/workflows/test.yml/badge.svg?branch=main)
[![Downloads](https://static.pepy.tech/badge/svs)](https://pepy.tech/project/svs)

- 🤔 What is SVS?
  - Semantic search via deep-learning vector embeddings.
  - A stupid-simple library for storing and retrieving your documents.

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
  - Supports storing arbitrary metadata with each document. 🗃️
  - Supports storing and querying (optional) parent-child relationships between documents. 👪
     - Fully hierarchical - parents can have parents, children can have children, whatever you need...
  - Both **sync** and **asyncio** implementations:
     - use the synchronous impl (`svs.KB`) for scripts, notebooks, etc
     - use the asyncio impl (`svs.AsyncKB`) for web-services, etc
  - 100% Python type hints!

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
- [Speed & Benchmarks](#speed-&-benchmarks)
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

Here is the _most simple_ use-case; it just queries a pre-built knowledge base!
This particular example queries a knowledge base of "Dad Jokes" 🤩.

(taken from [./examples/quickstart.py](./examples/quickstart.py))

```python
import svs   # <-- pip install -U svs

import os
from dotenv import load_dotenv; load_dotenv()
assert os.environ.get('OPENAI_API_KEY'), "You must set your OPENAI_API_KEY environment variable!"

#
# The database remembers which embeddings provider (e.g. OpenAI) was used.
#
# The "Dad Jokes" database below uses OpenAI embeddings, so that's why you had
# to set your OPENAI_API_KEY above!
#
# NOTE: The first time you run this script it will download this database,
#       so expect that to take a few seconds...
#
DB_URL = 'https://github.com/Rhobota/svs/raw/main/examples/dad_jokes/dad_jokes.sqlite.gz'


def demo() -> None:
    kb = svs.KB(DB_URL)

    records = kb.retrieve('chicken', n = 10)

    for record in records:
        score = record['score']
        text = record['doc']['text']
        print(f" 😆 score={score:.4f}: {text}\n")

    kb.close()


if __name__ == '__main__':
    demo()
```

⚠️ **Want to see how that _Dad Jokes_ knowledge base was created?** See: [./examples/dad_jokes/Build Dad Jokes KB.ipynb](<./examples/dad_jokes/Build Dad Jokes KB.ipynb>)

## Speed & Benchmarks

SQLite and NumPy are fast, thus SVS is fast 🏎️. Our goal is to minimize the amount of work done at the Python-layer.

Also, your bottleneck will *likely* be the remote API calls to get document embeddings (e.g. calling out to OpenAI to get embeddings will be the _slowest_ thing), so it's likely not critical to further optimize the Python-layer bits.

The following benchmarks were performed on 2018-era commodity hardware (Intel i3-8100):

| Dataset Size (# of documents)   | Load into SQLite | Obtain Embeddings (remote API call) | Compute Cosine Similarity + Sort + Retrieve Top-100 Documents § |
| ------------------------------- | ---------------- | ----------------------------------- | --------------------------------------------------------------- |
| 10,548 short jokes †            | 0.07 seconds     | 80 seconds                          | 0.5 seconds (first query) + 0.011 seconds (subsequent queries)  |
| 1,000,000 synthetic documents ‡ | 8 seconds        | 2 hours ¶                           | 2 minutes (first query) + 0.24 seconds (subsequent queries)     |

† Dad jokes database from [this notebook](<./examples/dad_jokes/Build Dad Jokes KB.ipynb>)  
‡ these one million synthetic documents have an average length of 1,200 characters, see [this notebook](<./examples/One Million Documents Benchmark.ipynb>)  
§ this time does _not_ include the time it takes to obtain the query string's embedding from the external service (i.e. from OpenAI); rather, it includes the time it takes to compute the cosine similarity with the query string and _all_ the documents (where embedding dimensionality is 1,536), then sort those results, and then retrieve the top-100 documents from the database; the first query is slow because it must load the vectors from disk into RAM; subsequent queries are _fast_ since those vectors stay cached in RAM  
¶ this is an estimate based on typical response times from OpenAI's embeddings API; for this test, we generate synthetic embeddings with dimensionality 1,536 to simulate the correct datasize and computation requirements

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
