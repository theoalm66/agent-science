"""Keeps user-supplied endpoints from being used to reach the machine's own network.

When guests may enter their own API endpoint, Pandora makes the request, from inside the
server's network (a Docker host in someone's home LAN, with Tailscale). Without a guard a guest
could point it at 127.0.0.1, 192.168.x.x (the vLLM server, the router), Tailscale addresses
(100.64.0.0/10) or a cloud metadata address. Two layers:

* check_public_endpoint(): cheap validation when settings are saved or a run starts, so the
  guest gets a clear message.
* SafeTransport: validates again for every request and connects to the address it validated, so a
  DNS answer that changes in between (DNS rebinding) cannot redirect the request.

Hosts the administrator lists in `server.guest_allowed_hosts` are exempt, e.g. a university API
that is only reachable on an internal address.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Callable, Iterable
from urllib.parse import urlsplit

import httpx

Resolver = Callable[..., list]


class EndpointNotAllowed(ValueError):
    pass


def _check_address(ip_text: str, host: str) -> None:
    ip = ipaddress.ip_address(ip_text.split("%", 1)[0])  # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global:
        raise EndpointNotAllowed(
            f"{host} points to a private or internal address ({ip}). Only public endpoints can be used; "
            "ask the Pandora administrator to allow this host."
        )


def is_allowed(allowed: set[str], host: str, port: int) -> bool:
    """`allowed` holds lower-case entries: `host` (any port) or `host:port` (only that port)."""
    host = host.lower()
    return host in allowed or f"{host}:{port}" in allowed


def resolve_public(host: str, port: int, resolver: Resolver = socket.getaddrinfo) -> list[str]:
    """Addresses for `host`, all of which must be public. Raises EndpointNotAllowed otherwise."""
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        try:
            infos = resolver(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            raise EndpointNotAllowed(f"Cannot resolve {host}: {e}") from e
        addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise EndpointNotAllowed(f"{host} has no address")
    for address in addresses:
        _check_address(address, host)
    return addresses


def check_public_endpoint(
    url: str,
    allowed_hosts: Iterable[str] = (),
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    """Validate an endpoint URL typed by a guest and return it normalised (no trailing slash)."""
    url = (url or "").strip()
    parts = urlsplit(url)
    host = parts.hostname or ""
    allowed = {h.strip().lower() for h in allowed_hosts}
    if parts.scheme not in ("https", "http") or not host:
        raise EndpointNotAllowed("The endpoint must be a full URL such as https://api.example.com/v1")
    if parts.username or parts.password:
        raise EndpointNotAllowed("Put the key into the API key field, not into the URL")
    if parts.query or parts.fragment:
        raise EndpointNotAllowed("The endpoint URL must not contain a query string or fragment")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as e:
        raise EndpointNotAllowed("The endpoint URL has an invalid port") from e
    if is_allowed(allowed, host, port):
        return url.rstrip("/")
    if parts.scheme != "https":
        raise EndpointNotAllowed("The endpoint must use https, otherwise your key would be sent unencrypted")
    resolve_public(host, port, resolver)
    return url.rstrip("/")


class SafeTransport(httpx.BaseTransport):
    """httpx transport that only connects to public addresses and pins the validated one."""

    def __init__(
        self,
        allowed_hosts: Iterable[str] = (),
        inner: httpx.BaseTransport | None = None,
        resolver: Resolver = socket.getaddrinfo,
    ):
        self.allowed = {h.strip().lower() for h in allowed_hosts}
        self.inner = inner or httpx.HTTPTransport()
        self.resolver = resolver

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        if is_allowed(self.allowed, host, port):
            return self.inner.handle_request(request)
        address = resolve_public(host, port, self.resolver)[0]
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = host  # TLS name and certificate check use the real host name
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=address),
            headers=request.headers,  # keeps the original Host header
            stream=request.stream,
            extensions=extensions,
        )
        return self.inner.handle_request(pinned)

    def close(self) -> None:
        self.inner.close()


def safe_client(allowed_hosts: Iterable[str] = (), timeout: float = 600) -> httpx.Client:
    """A client for guest-supplied endpoints: public addresses only, no redirects."""
    return httpx.Client(transport=SafeTransport(allowed_hosts), timeout=timeout, follow_redirects=False)
