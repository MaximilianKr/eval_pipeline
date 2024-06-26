"""
This module processes the input data from a CSV file and creates a corpus
with cleaned and structured data for event plausibility analysis.
"""

import pandas as pd

df = pd.read_csv("clean_DTFit_human_dat.csv")


def create_corpus(corpus_csv: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the DataFrame to create a structured corpus.

    Args:
        df (pd.DataFrame): The input data frame containing the raw data.

    Returns:
        pd.DataFrame: A DataFrame containing the structured corpus.
    """
    data = []
    for item_num in sorted(corpus_csv.ItemNum.unique()):
        rows = corpus_csv[corpus_csv.ItemNum == item_num]
        good = rows[rows.Plausibility == "Plausible"].squeeze()
        bad = rows[rows.Plausibility == "Implausible"].squeeze()

        good_sentence = good.Sentence
        bad_sentence = bad.Sentence
        prefix = " ".join(good_sentence.split(" ")[:-1])
        good_continuation = good_sentence.split(" ")[-1].replace(".", "")
        bad_continuation = bad_sentence.split(" ")[-1].replace(".", "")

        data.append(
            {
                "item_id": item_num,
                "prefix": prefix,
                "good_continuation": good_continuation,
                "bad_continuation": bad_continuation,
                "category": "event_plausibility",
                "good_human_score": good.Score,
                "bad_human_score": bad.Score,
            }
        )

    return pd.DataFrame(data)


clean_df = create_corpus(df)
clean_df.to_csv("corpus.csv", index=False)
