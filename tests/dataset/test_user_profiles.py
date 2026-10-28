import pytest

from src.dataset.user_profiles import find_declarations, link_domain


@pytest.mark.parametrize("body,expected", [
    ("I voted for Trump and I'd do it again", [("vote", "right")]),
    ("I'm voting for Hillary Clinton", [("vote", "left")]),
    ("i supported Bernie in the primary.", [("vote", "left")]),
    ("I voted for Gary Johnson", [("vote", "third")]),
    ("As a Republican, I hate this", [("identity", "right")]),
    ("as a progressive I disagree", [("identity", "left")]),
    ("I'm not voting for Trump", []),
    ("I would never say I voted for Clinton", []),
    ("If I voted for Trump you'd be mad", []),
    ("I didn't say I supported Stein", []),
    ("> I voted for Trump\nthat's what he said", []),
    ("&gt; as a Democrat I think\nreply", []),
    ("Not sure. I voted for Stein.", [("vote", "third")]),
    ("I voted for him", []),
])
def test_find_declarations(body, expected):
    assert [(k, l) for k, l, _ in find_declarations(body)] == expected


@pytest.mark.parametrize("url,domain", [
    ("https://www.nytimes.com/2016/a", "nytimes.com"),
    ("http://m.huffingtonpost.com/x", "huffingtonpost.com"),
    ("https://i.imgur.com/a.jpg", "imgur.com"),
    ("https://www.bbc.co.uk/news", "bbc.co.uk"),
    ("https://np.reddit.com/r/politics", "reddit.com"),
    ("https://www.reddit\.com/r/x", "reddit.com"),
    ("http://", None),
])
def test_link_domain(url, domain):
    assert link_domain(url) == domain
