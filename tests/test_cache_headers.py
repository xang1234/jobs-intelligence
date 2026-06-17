from src.api.app import is_cacheable_get


def test_cacheable_get_paths():
    assert is_cacheable_get("GET", "/api/overview", 200)
    assert is_cacheable_get("GET", "/api/stats", 200)
    assert is_cacheable_get("GET", "/api/skills/cloud", 200)
    assert is_cacheable_get("GET", "/api/skills/related/python", 200)
    assert is_cacheable_get("GET", "/api/trends/companies/DBS", 200)


def test_non_cacheable():
    assert not is_cacheable_get("POST", "/api/search", 200)  # not a GET
    assert not is_cacheable_get("GET", "/api/career-delta/x/detail", 200)  # not whitelisted
    assert not is_cacheable_get("GET", "/health", 200)  # liveness, keep fresh
    assert not is_cacheable_get("GET", "/api/overview", 503)  # only cache success
