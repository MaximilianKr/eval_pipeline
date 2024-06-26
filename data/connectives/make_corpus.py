"""
This module processes the input data from a CSV file and creates a corpus
with cleaned and structured data.
"""

import re
import pandas as pd


def clean_whitespace(text):
    """
    Cleans up extra whitespace in the input text.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv("english_stimuli_connector_2014_drenhaus_et_al.json", delimiter="\t")

collected_data = []

for i in range(0, len(df["id"]), 2):
    row1 = df.iloc[i]
    row2 = df.iloc[i + 1] if i + 1 < len(df) else None

    sentence1 = clean_whitespace(row1["sentence"])
    prefix1, suffix1 = sentence1.split("[MASK]")
    prefix1 = clean_whitespace(prefix1)
    suffix1 = clean_whitespace(suffix1)
    good_continuation1 = f"{row1['target']} {suffix1}".strip().rstrip(".")
    bad_continuation1 = (
        f"{row2['target']} {suffix1}".strip().rstrip(".") if row2 is not None else ""
    )

    collected_data.append(
        {
            "item_id": row1["id"],
            "prefix": prefix1,
            "good_continuation": good_continuation1,
            "bad_continuation": bad_continuation1,
            "category": row1["connective_type"],
        }
    )

    if row2 is not None:
        sentence2 = clean_whitespace(row2["sentence"])
        prefix2, suffix2 = sentence2.split("[MASK]")
        prefix2 = clean_whitespace(prefix2)
        suffix2 = clean_whitespace(suffix2)
        good_continuation2 = f"{row2['target']} {suffix2}".strip().rstrip(".")
        bad_continuation2 = f"{row1['target']} {suffix2}".strip().rstrip(".")

        collected_data.append(
            {
                "item_id": row2["id"],
                "prefix": prefix2,
                "good_continuation": good_continuation2,
                "bad_continuation": bad_continuation2,
                "category": row2["connective_type"],
            }
        )

processed_df = pd.DataFrame(collected_data)
processed_df.to_csv("corpus.csv", index=False)
