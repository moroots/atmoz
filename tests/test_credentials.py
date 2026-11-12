import os
import stat
import sys

import pytest

from atmoz.exceptions import MissingCredentialError
from atmoz.resources import credentials as cred


def test_missing_secret_raises_with_instructions():
    with pytest.raises(MissingCredentialError, match="ATMOZ_AIRNOW_API_KEY"):
        cred.get_secret("airnow")
    assert cred.get_secret("airnow", required=False) is None


def test_set_user_secret_roundtrip(isolated_home):
    path = cred.set_secret("airnow", "k-123")
    assert path == isolated_home / ".atmoz.env"
    assert cred.get_secret("airnow") == "k-123"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_secret_file_is_private():
    path = cred.set_secret("airnow", "k-123")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_precedence_explicit_then_env_then_project_then_user(monkeypatch, tmp_path):
    cred.set_secret("airnow", "from-user")
    assert cred.get_secret("airnow") == "from-user"
    cred.set_secret("airnow", "from-project", scope="project")
    assert cred.get_secret("airnow") == "from-project"
    monkeypatch.setenv("ATMOZ_AIRNOW_API_KEY", "from-env")
    assert cred.get_secret("airnow") == "from-env"
    assert cred.get_secret("airnow", explicit="from-arg") == "from-arg"


def test_project_file_found_from_subdirectory(tmp_path, monkeypatch):
    cred.set_secret("airnow", "proj", scope="project")
    sub = tmp_path / "work" / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert cred.get_secret("airnow") == "proj"


def test_set_preserves_other_keys_and_quotes_are_parsed(tmp_path):
    target = tmp_path / "custom.env"
    target.write_text('# c\nOTHER="x y"\nexport ATMOZ_AIRNOW_API_KEY=old\n')
    cred.set_secret("airnow", "new", path=target)
    values = cred._parse(target)
    assert values == {"OTHER": "x y", "ATMOZ_AIRNOW_API_KEY": "new"}


def test_delete_secret():
    cred.set_secret("airnow", "k")
    assert cred.delete_secret("airnow") is True
    assert cred.get_secret("airnow", required=False) is None
    assert cred.delete_secret("airnow") is False


@pytest.mark.parametrize("bad", ["", "   ", "a\nb"])
def test_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        cred.set_secret("airnow", bad)


def test_env_file_override(monkeypatch, tmp_path):
    custom = tmp_path / "elsewhere.env"
    custom.write_text("ATMOZ_AIRNOW_API_KEY=override\n")
    monkeypatch.setenv("ATMOZ_ENV_FILE", str(custom))
    assert cred.get_secret("airnow") == "override"
