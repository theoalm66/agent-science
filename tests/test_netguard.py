import socket

import httpx
import pytest

from science_agent.netguard import EndpointNotAllowed, SafeTransport, check_public_endpoint, resolve_public


def resolver_for(*addresses):
    def resolve(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    return resolve


PUBLIC = resolver_for("93.184.216.34")


@pytest.mark.parametrize("url", [
    "https://192.168.2.84:8008/v1",          # the vLLM server on the home network
    "https://10.0.0.5/v1",
    "https://172.16.0.1/v1",
    "https://127.0.0.1:8765/v1",             # Pandora itself
    "https://[::1]/v1",
    "https://100.116.110.73/v1",             # Tailscale address (CGNAT range)
    "https://169.254.169.254/latest/meta-data",  # cloud metadata
    "https://[::ffff:10.0.0.1]/v1",          # IPv4 hidden in IPv6
    "https://0.0.0.0/v1",
])
def test_private_and_internal_addresses_are_rejected(url):
    with pytest.raises(EndpointNotAllowed, match="private or internal"):
        check_public_endpoint(url, resolver=PUBLIC)


def test_names_that_resolve_to_internal_addresses_are_rejected():
    with pytest.raises(EndpointNotAllowed, match="private or internal"):
        check_public_endpoint("https://innocent.example.com/v1", resolver=resolver_for("192.168.2.84"))
    # one internal address among public ones is enough to refuse
    with pytest.raises(EndpointNotAllowed):
        check_public_endpoint("https://mixed.example.com/v1", resolver=resolver_for("93.184.216.34", "10.1.2.3"))


@pytest.mark.parametrize("url, message", [
    ("http://api.example.com/v1", "https"),
    ("ftp://api.example.com/v1", "full URL"),
    ("api.example.com/v1", "full URL"),
    ("https://user:secret@api.example.com/v1", "API key field"),
    ("https://api.example.com/v1?key=abc", "query"),
    ("https://api.example.com:notaport/v1", "port"),
])
def test_malformed_or_unsafe_urls_are_rejected(url, message):
    with pytest.raises(EndpointNotAllowed, match=message):
        check_public_endpoint(url, resolver=PUBLIC)


def test_public_endpoints_are_accepted_and_normalised():
    assert check_public_endpoint(" https://api.openai.com/v1/ ", resolver=PUBLIC) == "https://api.openai.com/v1"
    assert check_public_endpoint("https://93.184.216.34:8443/v1", resolver=PUBLIC) == "https://93.184.216.34:8443/v1"


def test_unresolvable_host_is_rejected():
    def fail(*a, **k):
        raise socket.gaierror("nope")

    with pytest.raises(EndpointNotAllowed, match="Cannot resolve"):
        check_public_endpoint("https://does-not-exist.example/v1", resolver=fail)


def test_admin_allowlist_exempts_a_host_from_the_checks():
    internal = resolver_for("10.20.30.40")
    url = "http://api.hochschule.example/v1"
    with pytest.raises(EndpointNotAllowed):
        check_public_endpoint(url, resolver=internal)
    assert check_public_endpoint(url, ["API.hochschule.example"], resolver=internal) == url


def test_resolve_public_returns_the_validated_addresses():
    assert resolve_public("api.example.com", 443, resolver_for("93.184.216.34", "93.184.216.35")) == [
        "93.184.216.34", "93.184.216.35"]


def _transport(seen, resolver=PUBLIC, allowed=()):
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    return SafeTransport(allowed, inner=httpx.MockTransport(handler), resolver=resolver)


def test_requests_connect_to_the_validated_address_and_keep_the_hostname_for_tls():
    seen = []
    client = httpx.Client(transport=_transport(seen))
    assert client.post("https://api.example.com:8443/v1/chat/completions", json={"a": 1}).status_code == 200
    request = seen[0]
    assert request.url.host == "93.184.216.34" and request.url.port == 8443
    assert request.headers["host"] == "api.example.com:8443"
    assert request.extensions["sni_hostname"] == "api.example.com"
    assert b'"a"' in request.read()  # the body is passed on


def test_a_dns_answer_that_turns_internal_is_caught_on_every_request():
    answers = iter([["93.184.216.34"], ["192.168.2.84"]])
    seen = []
    transport = _transport(seen, resolver=lambda host, port, type=None: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in next(answers)])
    client = httpx.Client(transport=transport)
    assert client.get("https://rebind.example.com/v1/models").status_code == 200
    with pytest.raises(EndpointNotAllowed):
        client.get("https://rebind.example.com/v1/models")
    assert len(seen) == 1  # the second request never left


def test_ip_literals_to_internal_hosts_are_blocked_even_without_a_name():
    seen = []
    client = httpx.Client(transport=_transport(seen))
    with pytest.raises(EndpointNotAllowed):
        client.get("https://192.168.2.84:8008/v1/models")
    assert seen == []


def test_allowed_hosts_pass_through_untouched():
    seen = []
    client = httpx.Client(transport=_transport(seen, resolver=resolver_for("10.0.0.9"), allowed=["uni.example"]))
    client.get("http://uni.example/v1/models")
    assert seen[0].url.host == "uni.example" and "sni_hostname" not in seen[0].extensions


def test_redirects_are_not_followed():
    from science_agent.netguard import safe_client

    client = safe_client()
    assert client.follow_redirects is False
    client.close()


def test_allowed_entry_with_port_only_exempts_that_port():
    internal = resolver_for("192.168.2.84")
    url = "http://192.168.2.84:8008/v1"
    assert check_public_endpoint(url, ["192.168.2.84:8008"], resolver=internal) == url
    with pytest.raises(EndpointNotAllowed):
        check_public_endpoint("http://192.168.2.84:22/v1", ["192.168.2.84:8008"], resolver=internal)
    seen = []
    client = httpx.Client(transport=_transport(seen, resolver=internal, allowed=["192.168.2.84:8008"]))
    assert client.get("http://192.168.2.84:8008/v1/models").status_code == 200
    with pytest.raises(EndpointNotAllowed):
        client.get("http://192.168.2.84:2375/version")
    assert len(seen) == 1
