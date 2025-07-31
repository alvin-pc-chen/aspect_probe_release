import pandas as pd


def make_clean_data(infile: str, outfile: str):
    """
    Creates a cleaned version of the input data in `.tsv` format where edge
    case verbs that cannot be easily identified in the sentence are removed.
    Duplicate sentences (i.e. sentences with multiple verbs) are also removed.
    Also adds the `start_char` and `end_char` positions for downstream processing.

    :param str infile: Path to input file. Needs to be in `.tsv` format with
        columns "Sentence", "Verb", and "Label".
    :param str outfile: Path to output file in `.tsv` format.
    """

    df = pd.read_csv(infile, sep="\t")
    df = df[~df["Sentence"].duplicated(keep=False)]
    df = df.rename(
        columns={
            "Sentence": "sentence",
            "Verb": "verb",
            "Label": "label",
        }
    )

    def _sent_char_span(row):
        sent = row["sentence"]
        verb = row["verb"]
        start_char = 0
        end_char = 0
        for word in sent.split():
            check_word = "".join([c for c in word if c.isalpha()])
            if check_word == verb:
                end_char = start_char + len(word) - 1
                break
            start_char += len(word) + 1
        start, end = (start_char, end_char) if end_char > start_char else (0, 0)
        return pd.Series({"start_char": start, "end_char": end})

    df[["start_char", "end_char"]] = df.apply(_sent_char_span, axis=1)
    df = df[(df["start_char"] != 0) & (df["end_char"] != 0)]

    df[["verb", "label", "sentence", "start_char", "end_char"]].to_csv(
        outfile, sep="\t", index=False
    )
