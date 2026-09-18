"""Security policy primitives for fetching remote documents.

The :class:`RemoteFetchPolicy` is consulted for *every* DNS resolution and
*every* redirect hop while fetching a remote document:

* the URL ``scheme`` must be on an allow list (only ``http``/``https`` by
  default);
* the host is normalised (case folding, trailing-dot removal, IDNA encoding)
  and user information embedded in the authority is rejected;
* every resolved IP address is classified and checked against the restricted
  network table, which by default blocks loopback, link-local, private,
  carrier-grade NAT, multicast, unspecified, reserved and documentation
  ranges;
* Unix domain socket connections (including "unix proxies") are refused and
  environment-supplied proxies are ignored (enforced by the fetcher in
  :mod:`frogmouth.security.guarded_fetch`).

Streaming reads are bounded by limits on header bytes, compressed body
bytes, decompressed bytes (decompression-bomb protection), average read
rate, a total deadline and an allow list of content encodings and text
charsets.

Each fetch produces a :class:`FetchReport` describing every hop, the
addresses the hosts resolved to, the policy decision taken for each of them
and the transfer statistics; callers render this as redirect provenance
for the user.

A host that is blocked purely because it resolves to a restricted network
may be exempted by the user, but only via an explicit asynchronous prompt;
:class:`RemoteFetchPolicy` itself never persists anything -- the caller is
responsible for durably recording the exemption.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    ClassVar,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import httpx

# A v4 or v6 IP address (the stdlib's ``_BaseAddress`` is intentionally not
# part of the public typing surface).
IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RemoteFetchError(Exception):
    """Base class for errors raised while performing a guarded fetch."""

    report: Optional["FetchReport"] = None
    """The partial provenance report, attached once a fetch is in flight."""


class PolicyViolation(RemoteFetchError):
    """Raised when a fetch (or one of its hops) violates the fetch policy."""


class SchemeBlocked(PolicyViolation):
    """Raised when a URL uses a scheme that is not on the allow list."""


class BadRequestTarget(PolicyViolation):
    """Raised when a URL target is malformed or carries forbidden data."""


class HostBlocked(PolicyViolation):
    """Raised when a host resolves (only) to restricted network addresses.

    Attributes:
        host: The normalised host that was blocked.
        categories: The restricted network categories that were hit.
        addresses: ``(ip address, categories)`` pairs for every resolved
            address, for display in the user prompt.
        reason: A short human-readable explanation.
    """

    def __init__(
        self,
        host: str,
        categories: Tuple[str, ...],
        addresses: Tuple[Tuple[str, Tuple[str, ...]], ...],
        reason: str,
    ) -> None:
        self.host = host
        self.categories = categories
        self.addresses = addresses
        self.reason = reason
        detail = ", ".join(
            f"{address} ({'/'.join(labels)})" for address, labels in addresses
        )
        super().__init__(f"{reason}: host '{host}' resolved to {detail}.")


class UnixProxyBlocked(PolicyViolation):
    """Raised when a Unix domain socket connection is attempted."""


class DnsResolutionError(RemoteFetchError):
    """Raised when a host name cannot be resolved."""


class TransferLimitExceeded(PolicyViolation):
    """Raised when a header/body/decompression byte limit is exceeded."""


class RateLimitExceeded(PolicyViolation):
    """Raised when a stream is read faster than the policy allows."""


class DeadlineExceeded(PolicyViolation):
    """Raised when the overall fetch deadline passes."""


class UnsupportedEncoding(PolicyViolation):
    """Raised when the content encoding is not on the allow list."""


class UnsupportedCharset(PolicyViolation):
    """Raised when the text charset is not on the allow list."""


class RedirectLimitExceeded(PolicyViolation):
    """Raised when too many redirects are followed."""


# ---------------------------------------------------------------------------
# Network classification
# ---------------------------------------------------------------------------


# Labels used for the well-known restricted ranges. Keeping our own table
# (rather than relying solely on ``ipaddress`` flags) makes the decision
# identical on every supported Python version.
_NETWORK_TABLE: Tuple[
    Tuple[str, ipaddress.IPv4Network | ipaddress.IPv6Network], ...
] = (
    ("loopback", ipaddress.ip_network("127.0.0.0/8")),
    ("loopback", ipaddress.ip_network("::1/128")),
    ("unspecified", ipaddress.ip_network("0.0.0.0/8")),
    ("unspecified", ipaddress.ip_network("::/128")),
    ("link-local", ipaddress.ip_network("169.254.0.0/16")),
    ("link-local", ipaddress.ip_network("fe80::/10")),
    ("private", ipaddress.ip_network("10.0.0.0/8")),
    ("private", ipaddress.ip_network("172.16.0.0/12")),
    ("private", ipaddress.ip_network("192.168.0.0/16")),
    ("unique-local", ipaddress.ip_network("fc00::/7")),
    ("cgnat", ipaddress.ip_network("100.64.0.0/10")),
    ("ipv6-to-ipv4-relay", ipaddress.ip_network("192.88.99.0/24")),
    ("benchmarking", ipaddress.ip_network("198.18.0.0/15")),
    ("documentation", ipaddress.ip_network("192.0.2.0/24")),
    ("documentation", ipaddress.ip_network("198.51.100.0/24")),
    ("documentation", ipaddress.ip_network("203.0.113.0/24")),
    ("documentation-ipv6", ipaddress.ip_network("2001:db8::/32")),
    ("private-protocol", ipaddress.ip_network("192.0.0.0/24")),
    ("multicast", ipaddress.ip_network("224.0.0.0/4")),
    ("multicast", ipaddress.ip_network("ff00::/8")),
    ("reserved", ipaddress.ip_network("240.0.0.0/4")),
)

DEFAULT_BLOCKED_CATEGORIES: Tuple[str, ...] = (
    "loopback",
    "unspecified",
    "link-local",
    "private",
    "unique-local",
    "cgnat",
    "ipv6-to-ipv4-relay",
    "benchmarking",
    "documentation",
    "documentation-ipv6",
    "private-protocol",
    "multicast",
    "reserved",
    "restricted",
)
"""Network categories that are blocked unless the host is explicitly trusted."""


def classify_ip(  # pylint:disable=too-many-return-statements
    address: IPAddress,
) -> Tuple[str, ...]:
    """Classify an IP address into zero or more restricted categories.

    Args:
        address: The IP address to classify.

    Returns:
        A tuple of category labels; an empty tuple means the address is a
        normal public address.
    """
    # IPv4-mapped (or compatible) IPv6 addresses are classified using the
    # embedded IPv4 address -- ::ffff:127.0.0.1 must be treated as loopback.
    mapped: Optional[ipaddress.IPv4Address] = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    labels = [label for label, network in _NETWORK_TABLE if address in network]
    if address.is_multicast and "multicast" not in labels:
        labels.append("multicast")
    if address.is_unspecified and "unspecified" not in labels:
        labels.append("unspecified")
    if address.is_loopback and "loopback" not in labels:
        labels.append("loopback")
    if address.is_link_local and "link-local" not in labels:
        labels.append("link-local")
    if address.is_reserved and "reserved" not in labels:
        labels.append("reserved")
    if not labels and not address.is_global:
        # Anything else that the standard library considers non-global is
        # restricted by default (fail closed).
        labels.append("restricted")
    return tuple(labels)


def normalize_host(raw_host: str | bytes) -> str:
    """Normalise a URL host for comparison and persistence.

    Args:
        raw_host: The host as taken from a URL (ASCII or unicode).

    Returns:
        The lower-cased, trailing-dot-stripped, IDNA-encoded host.

    Raises:
        BadRequestTarget: If the host cannot be safely normalised.
    """
    if isinstance(raw_host, bytes):
        try:
            host = raw_host.decode("ascii")
        except UnicodeDecodeError as error:
            raise BadRequestTarget(
                "URL host contains non-ASCII bytes that are not IDNA encoded"
            ) from error
    else:
        host = raw_host
    host = host.strip().lower()
    if host.endswith("."):
        host = host[:-1]
    if not host:
        raise BadRequestTarget("URL does not contain a host name")
    # httpcore/httpx hand us IPv6 authorities without brackets, but be
    # defensive in case a caller provides the bracketed form.
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if " " in host or "%" in host:
        raise BadRequestTarget("URL host contains whitespace or a scope identifier")
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise BadRequestTarget(
                f"URL host {host!r} could not be IDNA encoded"
            ) from error
    return host


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyLimits:  # pylint:disable=too-many-instance-attributes
    """The numeric and allow-list limits enforced by the fetch policy."""

    max_header_bytes: int = 64 * 1024
    """Maximum total size, in bytes, of a response's header block."""

    max_body_bytes: int = 5 * 1024 * 1024
    """Maximum number of compressed (wire) body bytes that will be read."""

    max_decoded_bytes: int = 20 * 1024 * 1024
    """Maximum number of decompressed body bytes that will be accepted."""

    max_rate_bytes_per_sec: int = 1024 * 1024
    """Maximum sustained read rate in bytes per second (0 disables pacing)."""

    deadline_seconds: float = 30.0
    """The wall-clock deadline for the whole fetch, in seconds."""

    connect_timeout: float = 10.0
    """The timeout for establishing a TCP connection, in seconds."""

    read_timeout: float = 15.0
    """The timeout for waiting on a single read, in seconds."""

    write_timeout: float = 15.0
    """The timeout for sending request data, in seconds."""

    max_redirects: int = 5
    """The maximum number of redirect hops that will be followed."""

    read_chunk_size: int = 64 * 1024
    """The chunk size used while streaming a response body."""

    allowed_schemes: Tuple[str, ...] = ("http", "https")
    """The URL schemes that may be fetched."""

    allowed_encodings: Tuple[str, ...] = ("identity", "gzip", "deflate")
    """The response content encodings that may be decoded."""

    allowed_charsets: Tuple[str, ...] = ("utf-8", "us-ascii", "iso-8859-1")
    """The charsets that text responses may declare."""

    blocked_categories: Tuple[str, ...] = DEFAULT_BLOCKED_CATEGORIES
    """The IP network categories that are blocked for untrusted hosts."""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class ConnectionInfo:
    """A single resolved address considered while establishing a hop."""

    address: str
    """The literal IP address."""

    categories: Tuple[str, ...]
    """The restricted-network categories the address belongs to."""

    chosen: bool = False
    """Whether the connection was pinned to this address."""

    trusted_bypass: bool = False
    """Whether the address was only reachable thanks to a trusted host."""


@dataclass
class HopReport:  # pylint:disable=too-many-instance-attributes
    """A record of one request hop in a fetch."""

    hop: int
    url: str
    method: str
    host: str
    connections: list[ConnectionInfo] = field(default_factory=list)
    status_code: Optional[int] = None
    reason_phrase: str = ""
    redirect_to: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    trusted: bool = False


@dataclass
class FetchReport:  # pylint:disable=too-many-instance-attributes
    """The policy decisions and transfer statistics for a whole fetch."""

    requested_url: str
    hops: list[HopReport] = field(default_factory=list)
    final_url: Optional[str] = None
    header_bytes: int = 0
    body_bytes: int = 0
    decoded_bytes: int = 0
    content_encoding: str = "identity"
    charset: str = "utf-8"
    content_type: str = ""
    duration: float = 0.0
    rate_bytes_per_sec: float = 0.0
    trusted_hosts: list[str] = field(default_factory=list)

    @staticmethod
    def _human_bytes(value: float) -> str:
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024.0 or unit == "GiB":
                return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} GiB"

    def _hop_line(self, hop: HopReport) -> str:
        if hop.connections:
            preferred = next(
                (conn for conn in hop.connections if conn.chosen),
                hop.connections[0],
            )
            categories = "/".join(preferred.categories) or "public"
            target = f"{preferred.address} ({categories})"
        else:
            target = "no connection"
        bits = [f"`{hop.host}` -> {target}"]
        if hop.status_code is not None:
            bits.append(f"{hop.status_code} {hop.reason_phrase}".strip())
        if hop.redirect_to is not None:
            bits.append(f"redirect to `{hop.redirect_to}`")
        if hop.trusted:
            bits.append("trusted host exemption (persisted)")
        line = f"> **Hop {hop.hop + 1}** " + " · ".join(bits)
        warnings = "".join(f"\n> ! {warning}" for warning in hop.warnings)
        return line + warnings

    def render_markdown(self) -> str:
        """Render the report as a Markdown blockquote banner.

        Returns:
            Markdown text suitable for prefixing the fetched document with.
        """
        lines = [f"> **Remote fetch policy** · `{self.requested_url}`"]
        lines.extend(self._hop_line(hop) for hop in self.hops)
        stats = (
            f"headers {self._human_bytes(self.header_bytes)} · "
            f"body {self._human_bytes(self.body_bytes)} · "
            f"decoded {self._human_bytes(self.decoded_bytes)} · "
            f"encoding {self.content_encoding} · charset {self.charset} · "
            f"{self.duration:.1f} s"
        )
        if self.rate_bytes_per_sec:
            stats += f" · {self._human_bytes(self.rate_bytes_per_sec)}/s"
        lines.append(f"> {stats}")
        return "\n".join(lines)

    def render_text(self) -> str:
        """Render the report as plain text for error dialogs.

        Returns:
            The report as plain text.
        """
        lines = [f"Remote fetch policy report for {self.requested_url}"]
        for hop in self.hops:
            lines.append(self._hop_line(hop).lstrip("> "))
        lines.append(
            f"headers {self.header_bytes} B, body {self.body_bytes} B, "
            f"decoded {self.decoded_bytes} B, encoding "
            f"{self.content_encoding}, {self.duration:.1f} s"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


# Called when the policy needs the user to explicitly approve a blocked
# host. The boolean result indicates approval, at which point the caller is
# expected to have persisted the exemption.
TrustPrompter = Callable[
    [HostBlocked],
    Awaitable[bool],
]


class RemoteFetchPolicy:
    """The policy used to decide what may be fetched and how."""

    DEFAULT_PORTS: ClassVar[dict[str, int]] = {"http": 80, "https": 443}

    def __init__(
        self,
        limits: Optional[PolicyLimits] = None,
        trusted_hosts: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialise the policy.

        Args:
            limits: The limits to enforce; sensible secure defaults are
                used when omitted.
            trusted_hosts: Hosts that have been explicitly exempted from
                the restricted-network rules. Values should already be
                normalised hosts.
        """
        self.limits: PolicyLimits = limits if limits is not None else PolicyLimits()
        self._trusted_hosts: set[str] = set(trusted_hosts or ())

    @property
    def trusted_hosts(self) -> Tuple[str, ...]:
        """The hosts currently exempted from network restrictions."""
        return tuple(sorted(self._trusted_hosts))

    def is_trusted(self, host: str) -> bool:
        """Has the given normalised host been explicitly exempted?

        Args:
            host: The normalised host to check.

        Returns:
            ``True`` if the host is trusted.
        """
        return host in self._trusted_hosts

    def trust(self, host: str) -> None:
        """Add a host to the in-memory trusted set.

        Note:
            Persisting the exemption is the caller's responsibility; the
            policy deliberately keeps no durable state of its own.

        Args:
            host: The normalised host to trust.
        """
        self._trusted_hosts.add(host)

    def revoke(self, host: str) -> None:
        """Remove a host from the in-memory trusted set.

        Args:
            host: The normalised host to revoke.
        """
        self._trusted_hosts.discard(host)

    def check_scheme(self, scheme: str) -> str:
        """Validate a URL scheme.

        Args:
            scheme: The scheme to validate.

        Returns:
            The normalised scheme.

        Raises:
            SchemeBlocked: If the scheme is not allowed.
        """
        normalised = scheme.lower()
        if normalised not in self.limits.allowed_schemes:
            allowed = ", ".join(self.limits.allowed_schemes)
            raise SchemeBlocked(
                f"URL scheme '{scheme}' is not allowed (allowed: {allowed})"
            )
        return normalised

    def check_target(self, url: httpx.URL) -> str:
        """Validate the shape of a request URL and return its host.

        Args:
            url: The URL that is about to be requested.

        Returns:
            The normalised host.

        Raises:
            SchemeBlocked: For disallowed schemes.
            BadRequestTarget: For malformed authorities or embedded
                credentials.
            HostBlocked: For IP-literal hosts in restricted ranges.
        """
        self.check_scheme(url.scheme)
        if url.userinfo:
            raise BadRequestTarget(
                "URLs containing user names or passwords are not allowed"
            )
        raw_host = url.raw_host
        if raw_host is None:
            raise BadRequestTarget("URL does not contain a host name")
        host = normalize_host(raw_host)
        # An IP-literal host can be classified immediately, without any
        # DNS involved.
        try:
            literal: Optional[IPAddress] = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            categories = classify_ip(literal)
            blocked = tuple(
                category
                for category in categories
                if category in self.limits.blocked_categories
            )
            if blocked and not self.is_trusted(host):
                raise HostBlocked(
                    host,
                    blocked,
                    ((str(literal), categories),),
                    "IP-literal URL target is on a restricted network",
                )
        return host

    async def resolve(
        self, host: str, port: int, timeout: Optional[float] = None
    ) -> Tuple[IPAddress, ...]:
        """Resolve a host name to all of its addresses.

        Args:
            host: The normalised host to resolve.
            port: The port being connected to.
            timeout: Maximum seconds to wait for DNS resolution.

        Returns:
            Every unique address returned by the resolver.

        Raises:
            DnsResolutionError: If the host cannot be resolved.
        """
        loop = asyncio.get_running_loop()
        try:
            infos = await asyncio.wait_for(
                loop.getaddrinfo(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                ),
                timeout=timeout or self.limits.connect_timeout,
            )
        except asyncio.TimeoutError as error:
            raise DnsResolutionError(
                f"DNS resolution for '{host}' timed out"
            ) from error
        except (socket.gaierror, UnicodeError, OSError) as error:
            raise DnsResolutionError(f"Could not resolve '{host}': {error}") from error
        addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
        if not addresses:
            raise DnsResolutionError(f"No DNS records returned for '{host}'")
        return tuple(sorted(addresses, key=str))

    def classify_resolution(
        self, addresses: Iterable[IPAddress]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Classify a set of resolved addresses.

        Args:
            addresses: The resolved addresses.

        Returns:
            ``(address, categories)`` pairs in resolution order.
        """
        return tuple((str(address), classify_ip(address)) for address in addresses)
