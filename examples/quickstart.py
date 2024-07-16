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
