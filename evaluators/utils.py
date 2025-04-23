import os
import pandas as pd
from tqdm import tqdm

from tabulate import tabulate

model_names = [
    'cogvlm', 'evovlm_jp_v1_7b', 'llava1_5',
    'llava1_6', 'qwenvl_chat', 'gpt4o'
]

def parse_multichoice_response(s):
    if type(s) != str:
        return "n/a"
    s = s.strip()
    if s[0] not in "ABCDEFGH":
        return "n/a"
    return s[0]

def circular_correctness_multichoice(df):
    expected = df.expected
    response = df.response.map(parse_multichoice_response)
    is_correct = (expected == response).all()
    return 1 if is_correct else 0

#====================================================
# Ensemble accuracy
#====================================================
def l_choices_to_choice_dict(l_choices):
    l_alphabet = list("ABCDEFG")
    l_text = l_choices.split(",")
    assert len(l_alphabet) >= len(l_text), l_text
    return dict(zip(l_alphabet, l_text))

def response_to_choice_text(l_choices, response):
    d = l_choices_to_choice_dict(l_choices)
    text = d.get(
        parse_multichoice_response(response),
        "n/a"
    )
    return text

def ensemble_correctness_multichoice(df):
    response_text = df.apply(lambda row: response_to_choice_text(row.l_choices, row.response), axis=1)
    expected_text = df.apply(lambda row: response_to_choice_text(row.l_choices, row.expected), axis=1)
    assert expected_text.nunique() == 1
    label = expected_text.unique()[0]

    # Calculate the frequency of each item
    frequency = response_text.value_counts()

    # Identify the item(s) with the maximum frequency
    max_frequency = frequency.max()
    most_frequent_items = frequency[frequency == max_frequency]

    # Check if there is only one item with the maximum frequency
    if len(most_frequent_items) == 1:
        most_frequent_item = most_frequent_items.index[0]
        # The result is correct if that item is the label
        return most_frequent_item == label
    else:
        return False


def get_latex_table(df):
    table = tabulate(
        df, headers='keys',
        tablefmt='latex', numalign='left', stralign='left',
        showindex=True,
        floatfmt='.3f'
    )
    table = table.replace("\\_", "-")
    return table

def debug():
    import ipdb;
    ipdb.set_trace()
