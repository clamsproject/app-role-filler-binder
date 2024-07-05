"""
Tests for `rfb` parser function
"""

import pytest
from utils.rfb import parse_sequence_tags


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # normal
        (
                [('FILL', 'Jane Doe'), ('ROLE', 'Dir., Health Services')],
                [{'Filler': 'Jane Doe', 'Role': 'Dir., Health Services'}]
        ),
        # empty role
        (
                [('FILL', 'Jane Doe')],
                [{'Filler': 'Jane Doe', 'Role': ''}]
        )
    ]
)
def test_parse_chyron(test_input, expected):
    """
    Tests the parser function on chyron ocr results.
    Args:
        test_input (list[tuple[str, str]]): A list of tuples containing name/role mentions and their labels.
        expected (list[dict[str, str]]): A list of dictionaries that pair each filler mention with the correct role.

    Returns: None

    """
    assert parse_sequence_tags(test_input, 'chyron') == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # normal
        (
                [('ROLE', 'Director'), ('FILL', 'Joe Bloggs'), ('ROLE', 'Technical Director'), ('FILL', 'Bill Taylor')],
                [{'Filler': 'Joe Bloggs', 'Role': 'Director'}, {'Filler': 'Bill Taylor', 'Role': 'Technical Director'}]
        ),
        # all empty role
        (
                [('FILL', 'Jane Doe'),
                 ('FILL', 'Richard Roe'),
                 ('FILL', 'Bill Taylor')],
                [{'Filler': 'Jane Doe', 'Role': ''},
                 {'Filler': 'Richard Roe', 'Role': ''},
                 {'Filler': 'Bill Taylor', 'Role': ''}]
        ),
        # some empty role
        (
                [('FILL', 'Jane Doe'),
                 ('ROLE', 'Audio'),
                 ('FILL', 'Richard Roe'),
                 ('FILL', 'Bill Taylor')],
                [{'Filler': 'Jane Doe', 'Role': ''},
                 {'Filler': 'Richard Roe', 'Role': 'Audio'},
                 {'Filler': 'Bill Taylor', 'Role': 'Audio'}]
        ),
        # empty filler
        (
                [('ROLE', 'Studio Engineers'),
                 ('FILL', 'Jane Doe'),
                 ('FILL', 'Richard Roe'),
                 ('ROLE', 'Audio')],
                [{'Filler': 'Jane Doe', 'Role': 'Studio Engineers'},
                 {'Filler': 'Richard Roe', 'Role': 'Studio Engineers'},
                 {'Filler': '', 'Role': 'Audio'}]
        ),
        # empty role and empty filler
        (
                [('FILL', 'Jane Doe'),
                 ('FILL', 'Richard Roe'),
                 ('ROLE', 'Audio')],
                [{'Filler': 'Jane Doe', 'Role': ''},
                 {'Filler': 'Richard Roe', 'Role': ''},
                 {'Filler': '', 'Role': 'Audio'}]
        )
    ]
)
def test_parse_credits(test_input, expected):
    """
    Tests the parser function on credits ocr results.
    Args:
        test_input (list[tuple[str, str]]): A list of tuples containing name/role mentions and their labels.
        expected (list[dict[str, str]]): A list of dictionaries that pair each filler mention with the correct role.

    Returns: None

    """
    assert parse_sequence_tags(test_input, 'credits') == expected
