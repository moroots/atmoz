from atmoz.resources.urls import build_url, redact


def test_build_url_joins_and_quotes_segments():
    assert build_url("https://x.org/api/", "data", 1) == "https://x.org/api/data/1"
    assert build_url("https://x.org", "a b", "c/d") == "https://x.org/a%20b/c%2Fd"


def test_build_url_skips_empty_segments_and_none_params():
    assert build_url("https://x.org", "", None, "ok", params={"a": None}) == "https://x.org/ok"


def test_build_url_params_lists_and_bools():
    url = build_url("https://x.org", params={"p": ["OZONE", "PM25"], "flag": True, "n": 3})
    assert url == "https://x.org?p=OZONE%2CPM25&flag=true&n=3"


def test_redact_masks_query_secrets_and_literals():
    text = "GET https://h/aq?startDate=1&API_KEY=abc123&x=2 failed; key abc123"
    out = redact(text, "abc123")
    assert "abc123" not in out
    assert "API_KEY=***" in out
    assert "startDate=1" in out


def test_redact_handles_url_encoded_secret():
    assert "a%2Fb" not in redact("token a%2Fb", "a/b")
