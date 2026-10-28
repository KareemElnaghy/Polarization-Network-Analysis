"""Row-level text exclusions shared by the dataset and prep-job code.

``REMOVED_TEXTS`` was defined in ``src/dataset/build_edges.py``; it lives here now so the same rule
is applied to comments and to edges. ``PRIVACY_OVERWRITE_REGEX`` is new: ``build_edges`` had no
privacy-script detector (it only flagged removed/deleted text).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

REMOVED_TEXTS = ["[removed]", "[deleted]"]
# One open-source script overwrites a user's comments with this text (about 2,000 copies a month).
PRIVACY_OVERWRITE_REGEX = r"(?i)^this comment has been overwritten by an open source script"


def removed_mask(text: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """True where the text is exactly ``[removed]`` or ``[deleted]``."""
    return pc.fill_null(pc.is_in(text, pa.array(REMOVED_TEXTS)), False)


def privacy_overwritten_mask(text: pa.ChunkedArray | pa.Array, regex: str = PRIVACY_OVERWRITE_REGEX):
    """True where the text is a privacy-script overwrite."""
    return pc.fill_null(pc.match_substring_regex(text, regex), False)


def unusable_text_mask(text, regex: str = PRIVACY_OVERWRITE_REGEX):
    """True where the text is removed, deleted or a privacy-script overwrite."""
    return pc.or_(removed_mask(text), privacy_overwritten_mask(text, regex))
