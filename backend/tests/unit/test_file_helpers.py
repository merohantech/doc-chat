import pytest

from app.services.context_builder import Attachment, user_message
from app.services.file_service import clean_filename, unique_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("C:\\Users\\ada\\report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ('bad<>:"|?*name.txt', "bad_______name.txt"),
        ("   ", "file"),
        ("x" * 300 + ".pdf", "x" * 251 + ".pdf"),
    ],
)
def test_clean_filename(raw: str, expected: str) -> None:
    assert clean_filename(raw) == expected


def test_unique_filename_skips_names_in_use() -> None:
    taken = {"report.pdf", "report (2).pdf"}

    assert unique_filename("report.pdf", taken) == "report (3).pdf"


def test_user_message_without_files_is_plain_text() -> None:
    assert user_message("Hi") == {"role": "user", "content": "Hi"}


def test_user_message_puts_files_first_and_question_last() -> None:
    message = user_message(
        "Summarise both.",
        [
            Attachment("a.txt", 1, "alpha"),
            Attachment('b "x".md', 2, "beta", replaces_sent_version=True),
        ],
    )

    assert message == {
        "role": "user",
        "content": [
            {"type": "text", "text": '<file name="a.txt" version="1">\nalpha\n</file>'},
            {
                "type": "text",
                "text": '<file name="b &quot;x&quot;.md" version="2" replaces="version 1">'
                "\nbeta\n</file>",
            },
            {"type": "text", "text": "Summarise both."},
        ],
    }
