import json
import typer
import spacy
from pathlib import Path
from typing import Dict, Union


Prediction = Dict[str, Union[str, float]]


class SentenceClassifier:
    """Pipe sentence through a spacy model and return categories.

    Usage
    -----
    >>> nlp = spacy.load("path/to/model")
    >>> snippet = '''
    Secretary of State Mike Pompeo and Defense Secretary Mark T. Esper met with
    President Trump at the White House following the attack. The president said
    he will make a statement Wednesday morning...
    '''
    >>> classifier = SentenceClassifier(nlp=nlp)
    >>> classifier.predict(snippet)
    """
    def __init__(self, nlp: spacy.language.Language) -> None:
        self.nlp = nlp

    def predict(self, sentence: str) -> Prediction:
        doc = self.nlp(sentence)
        return {"sentence": sentence, **doc.cats}


def write_jsonl(p: Path, line: Prediction) -> None:
    with p.open(mode="a", encoding="utf-8") as lines:
        json.dump(line, lines)
        lines.write("\n")

    
def check(
    model: Path = typer.Argument(..., exists=True, dir_okay=True),
    input_text: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_jsonl: Path = typer.Argument(..., dir_okay=False)
) -> None:
    nlp = spacy.load(model)
    classifier = SentenceClassifier(nlp)
    with input_text.open(mode="r", encoding="utf-8") as sentences:
        for sentence in sentences:
            write_jsonl(output_jsonl, classifier.predict(sentence))
    

if __name__ == "__main__":
    typer.run(check)