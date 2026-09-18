"""The policy-enforcing HTTP fetcher.

This module contains the runtime half of the remote fetch machinery: the
:class:`GuardedFetcher` performs a manual, hop-by-hop fetch, delegating
every scheme/host/IP decision to
:class:`~frogmouth.security.remote_fetch.RemoteFetchPolicy` while streaming
the response body under the configured byte, rate, deadline and encoding
limits.

Connections are made through a wrapper around httpcore's async network
backend which re-resolves and re-validates every host name itself and then
pins the socket to one of the addresses it just validated, closing the
DNS-rebinding gap between a check and a connect.
"""

from __future__ import annotations

import asyncio
import codecs
import ipaddress
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import httpx
from httpcore import AsyncNetworkBackend, AsyncNetworkStream

from .remote_fetch import (
    ConnectionInfo,
    DeadlineExceeded,
    FetchReport,
    HopReport,
    HostBlocked,
    IPAddress,
    PolicyViolation,
    RateLimitExceeded,
    RedirectLimitExceeded,
    RemoteFetchError,
    RemoteFetchPolicy,
    TransferLimitExceeded,
    TrustPrompter,
    UnixProxyBlocked,
    UnsupportedCharset,
    UnsupportedEncoding,
    normalize_host,
)

_ENCODING_ALIASES = {
    "x-gzip": "gzip",
    "gzip": "gzip",
    "x-deflate": "deflate",
    "deflate": "deflate",
    "identity": "identity",
    "": "identity",
}

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

_CHARSET_ALIASES = {
    "": "utf-8",
    "utf8": "utf-8",
    "ascii": "us-ascii",
    "latin1": "iso-8859-1",
    "latin-1": "iso-8859-1",
    "iso8859-1": "iso-8859-1",
}


@dataclass
class FetchResult:
    """The successful result of a guarded fetch."""

    text: str
    content_type: str
    final_url: httpx.URL
    status_code: int
    report: FetchReport


class _PolicyNetworkBackend(AsyncNetworkBackend):
    """Wrap httpcore's default backend, validating and pinning every connect.

    The wrapper implements the small ``AsyncNetworkBackend`` protocol shared
    by httpcore 0.17 and 1.x: ``connect_tcp``, ``connect_unix_socket`` and
    ``sleep``.
    """

    def __init__(self, inner: AsyncNetworkBackend, guard: "_FetchGuard") -> None:
        """Initialise the backend wrapper.

        Args:
            inner: The stock httpcore network backend used for the actual
                socket work.
            guard: The per-fetch guard holding the policy and report.
        """
        self._inner = inner
        self._guard = guard

    async def connect_tcp(  # pylint:disable=too-many-arguments,too-many-positional-arguments
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Optional[Iterable[Any]] = None,
        **kwargs: Any,
    ) -> AsyncNetworkStream:
        """Resolve, validate and pin the address before opening a socket.

        The guard re-resolves the name here, validates every answer and
        hands back a specific address to connect to; httpcore keeps using
        the *original* host for TLS SNI/certificate verification.
        """
        pinned = await self._guard.establish(host, port, timeout)
        return await self._inner.connect_tcp(
            pinned,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=list(socket_options or []),
            **kwargs,
        )

    async def connect_unix_socket(  # pylint:disable=unused-argument
        self,
        path: str,
        timeout: Optional[float] = None,
        socket_options: Optional[Iterable[Any]] = None,
    ) -> AsyncNetworkStream:
        """Refuse every Unix domain socket connection, including proxies."""
        raise UnixProxyBlocked(
            "Connections to Unix domain sockets (including Unix proxies) "
            "are blocked by the remote fetch policy"
        )

    async def sleep(self, seconds: float) -> None:
        """Defer to the wrapped backend's sleep helper."""
        await self._inner.sleep(seconds)


class _FetchGuard:
    """Per-fetch state shared between the redirect loop and the backend."""

    def __init__(self, policy: RemoteFetchPolicy) -> None:
        """Initialise the guard.

        Args:
            policy: The policy to consult for every decision.
        """
        self.policy = policy
        self.report = FetchReport(requested_url="")
        self.current_hop: Optional[HopReport] = None
        self.started: float = 0.0
        """Loop time at which the fetch started (set by the fetcher)."""
        self.deadline: float = 0.0
        """Absolute loop-time deadline for the whole fetch."""

    def begin_hop(self, hop: int, method: str, url: httpx.URL, host: str) -> HopReport:
        """Begin recording a new request hop.

        Args:
            hop: The zero-based hop number.
            method: The request method.
            url: The URL being requested.
            host: The provisional (raw) authority for the hop.

        Returns:
            The new hop report.
        """
        hop_report = HopReport(hop=hop, url=str(url), method=method, host=host)
        self.report.hops.append(hop_report)
        self.current_hop = hop_report
        return hop_report

    def remember_connection(self, info: ConnectionInfo) -> None:
        """Record an address for the current hop (de-duped within a hop).

        httpcore may retry a connect, so the same address can be reported
        more than once; de-duplication is intentionally scoped to a single
        hop so every redirect hop shows its own fresh resolution (the
        pool is configured not to reuse connections across requests).

        Args:
            info: The connection information to record.
        """
        if self.current_hop is None:
            return
        existing = next(
            (
                connection
                for connection in self.current_hop.connections
                if connection.address == info.address
            ),
            None,
        )
        if existing is None:
            self.current_hop.connections.append(info)
        elif info.chosen:
            existing.chosen = True
            existing.trusted_bypass = info.trusted_bypass

    def record_block(self, decision: HostBlocked) -> None:
        """Record addresses from a pre-connect block for provenance.

        Args:
            decision: The decision that blocked the hop.
        """
        if self.current_hop is None:
            return
        known = {connection.address for connection in self.current_hop.connections}
        for address, categories in decision.addresses:
            if address not in known:
                self.current_hop.connections.append(
                    ConnectionInfo(address=address, categories=categories)
                )

    async def establish(self, host: str, port: int, timeout: Optional[float]) -> str:
        """Resolve, validate and choose the address for a new connection.

        Args:
            host: The host httpcore is about to connect to.
            port: The port httpcore is about to connect to.
            timeout: The connect timeout hint.

        Returns:
            The literal IP address the socket must be opened to.
        """
        host = normalize_host(host)
        try:
            literal = ipaddress.ip_address(host)
            addresses: Tuple[IPAddress, ...] = (literal,)
        except ValueError:
            addresses = await self.policy.resolve(host, port, timeout)
        classified = self.policy.classify_resolution(addresses)
        trusted = self.policy.is_trusted(host)
        blocked: list[tuple[str, tuple[str, ...]]] = []
        allowed: list[str] = []
        for address, categories in classified:
            restricted = tuple(
                category
                for category in categories
                if category in self.policy.limits.blocked_categories
            )
            if restricted:
                blocked.append((address, categories))
            else:
                allowed.append(address)
            self.remember_connection(
                ConnectionInfo(
                    address=address,
                    categories=categories,
                    chosen=False,
                    trusted_bypass=trusted and bool(restricted),
                )
            )
        if blocked and not allowed and not trusted:
            categories = tuple(
                sorted({label for _, labels in blocked for label in labels})
            )
            raise HostBlocked(
                host,
                categories,
                tuple(blocked),
                "Restricted network destination blocked",
            )
        if blocked and not trusted:
            # Mixed answers: the underlying client is free to pick any
            # record, so fail closed rather than gamble on its choice.
            categories = tuple(
                sorted({label for _, labels in blocked for label in labels})
            )
            raise HostBlocked(
                host,
                categories,
                classified,
                "Host name returned a mix of public and restricted "
                "addresses; refusing to gamble on the resolver's choice",
            )
        chosen = allowed[0] if allowed else blocked[0][0]
        self.remember_connection(
            ConnectionInfo(
                address=chosen,
                categories=dict(classified)[chosen],
                chosen=True,
                trusted_bypass=trusted and not allowed,
            )
        )
        if trusted and self.current_hop is not None:
            self.current_hop.trusted = True
            if host not in self.report.trusted_hosts:
                self.report.trusted_hosts.append(host)
        return chosen


class _BodyDecoder:
    """Incrementally decode a streaming response body."""

    def __init__(self, encoding: str) -> None:
        """Initialise the decoder.

        Args:
            encoding: The content encoding to decode (identity/gzip/deflate).
        """
        self.encoding = encoding
        self._gzip = encoding == "gzip"
        if encoding == "gzip":
            self._decompressor: Optional[zlib._Decompress] = zlib.decompressobj(
                16 + zlib.MAX_WBITS
            )
        elif encoding == "deflate":
            self._decompressor = zlib.decompressobj()
        else:
            self._decompressor = None

    def feed(self, chunk: bytes) -> bytes:
        """Feed one wire chunk and return the decoded bytes.

        Args:
            chunk: One raw compressed chunk.

        Returns:
            The decoded output so far.
        """
        if self._decompressor is None:
            return chunk
        if not self._gzip:
            try:
                decoded: bytes = self._decompressor.decompress(chunk)
            except zlib.error as error:
                raise UnsupportedEncoding(
                    f"Could not decode '{self.encoding}' response body"
                ) from error
            return decoded
        # gzip may consist of multiple concatenated members.
        output = b""
        while chunk:
            try:
                output += self._decompressor.decompress(chunk)
            except zlib.error as error:
                raise UnsupportedEncoding(
                    "Could not decode 'gzip' response body"
                ) from error
            if not self._decompressor.eof:
                break
            chunk = self._decompressor.unused_data
            self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        return output

    def finish(self) -> bytes:
        """Flush the decoder and return any remaining output.

        Returns:
            Any trailing decoded bytes.
        """
        if self._decompressor is None or not self._gzip:
            tail = b"" if self._decompressor is None else self._decompressor.flush()
            if self._decompressor is not None:
                self._check_trailing(tail)
            return tail
        output = self._decompressor.flush()
        while self._decompressor.unused_data:
            chunk = self._decompressor.unused_data
            self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            output += self._decompressor.decompress(chunk)
            output += self._decompressor.flush()
        return output

    def _check_trailing(self, tail: bytes) -> None:
        if tail:
            raise UnsupportedEncoding(
                f"Trailing data while decoding '{self.encoding}' response body"
            )


class GuardedFetcher:  # pylint:disable=too-few-public-methods
    """Fetch remote documents while enforcing a :class:`RemoteFetchPolicy`."""

    def __init__(
        self,
        policy: RemoteFetchPolicy,
        *,
        user_agent: str,
        trust_prompter: Optional[TrustPrompter] = None,
    ) -> None:
        """Initialise the fetcher.

        Args:
            policy: The policy to enforce.
            user_agent: The User-Agent header to send.
            trust_prompter: An optional async callback invoked when a host
                is blocked because of its network category. Returning
                ``True`` exempts the host for this policy instance; the
                caller is expected to persist the exemption.
        """
        self.policy = policy
        self.user_agent = user_agent
        self.trust_prompter = trust_prompter

    def _guarded_transport(self, guard: _FetchGuard) -> httpx.AsyncHTTPTransport:
        # Build a stock transport, then replace the pool's network backend
        # with our validating wrapper. trust_env=False ensures no proxy,
        # .netrc or cookie configuration is picked up from the environment.
        # Keep-alive connections are disabled so that *every* hop (not
        # just the first one to an origin) performs a fresh DNS resolution
        # and IP validation.
        limits = httpx.Limits(max_keepalive_connections=0)
        transport = httpx.AsyncHTTPTransport(trust_env=False, limits=limits)
        # httpcore exposes the backend as a swappable pool attribute; the
        # wrapper keeps the same AsyncNetworkBackend protocol.
        # pylint:disable=protected-access
        pool = transport._pool
        pool._network_backend = _PolicyNetworkBackend(pool._network_backend, guard)
        # pylint:enable=protected-access
        return transport

    async def _prompt_for_trust(self, error: HostBlocked) -> bool:
        if self.trust_prompter is None:
            return False
        if not await self.trust_prompter(error):
            return False
        self.policy.trust(error.host)
        return True

    @staticmethod
    def _header_size(response: httpx.Response) -> int:
        return sum(len(name) + len(value) + 4 for name, value in response.headers.raw)

    @staticmethod
    def _parse_content_type(
        response: httpx.Response,
    ) -> tuple[str, str]:
        header = response.headers.get("content-type", "")
        media_type, _, params = header.partition(";")
        charset = ""
        for param in params.split(";"):
            name, sep, value = param.partition("=")
            if sep and name.strip().lower() == "charset":
                charset = value.strip().strip('"').lower()
        return media_type.strip().lower(), charset

    def _content_encoding(self, response: httpx.Response) -> str:
        raw = response.headers.get("content-encoding", "")
        tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
        normalised: list[str] = []
        for token in tokens:
            if token == "identity":
                continue
            if token not in _ENCODING_ALIASES:
                raise UnsupportedEncoding(f"Unsupported content encoding '{token}'")
            normalised.append(_ENCODING_ALIASES[token])
        if len(normalised) > 1:
            raise UnsupportedEncoding(
                f"Stacked content encodings ('{raw}') are not supported"
            )
        encoding = normalised[0] if normalised else "identity"
        if encoding not in self.policy.limits.allowed_encodings:
            raise UnsupportedEncoding(
                f"Content encoding '{encoding}' is not allowed by the policy"
            )
        return encoding

    def _decode_text(self, body: bytes, declared_charset: str) -> tuple[str, str]:
        charset = _CHARSET_ALIASES.get(declared_charset, declared_charset)
        if declared_charset:
            try:
                charset = codecs.lookup(charset).name
            except LookupError as error:
                raise UnsupportedCharset(
                    f"Unknown response charset '{declared_charset}'"
                ) from error
            allowed = {
                codecs.lookup(item).name for item in self.policy.limits.allowed_charsets
            }
            if charset not in allowed:
                raise UnsupportedCharset(
                    f"Response charset '{charset}' is not allowed by the policy"
                )
            try:
                return body.decode(charset), codecs.lookup(charset).name
            except UnicodeDecodeError as error:
                raise UnsupportedCharset(
                    f"Response body is not valid {charset}"
                ) from error
        try:
            return body.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            return body.decode("iso-8859-1"), "iso-8859-1"

    async def _pace_reading(self, wire_bytes: int, guard: _FetchGuard) -> None:
        """Sleep as needed so the wire read rate stays under the limit.

        Args:
            wire_bytes: The number of wire bytes read so far.
            guard: The per-fetch state holding timing information.
        """
        max_rate = self.policy.limits.max_rate_bytes_per_sec
        if not max_rate:
            return
        loop = asyncio.get_running_loop()
        expected = wire_bytes / max_rate
        elapsed = loop.time() - guard.started
        if expected <= elapsed:
            return
        delay = expected - elapsed
        if delay > guard.deadline - loop.time():
            raise RateLimitExceeded(
                "Pacing the response would exceed the fetch deadline"
            )
        await asyncio.sleep(delay)

    async def _read_body(
        self,
        response: httpx.Response,
        guard: _FetchGuard,
    ) -> tuple[bytes, str]:
        """Stream the response body, enforcing every transfer limit.

        Args:
            response: The response whose body is streamed.
            guard: The per-fetch state holding the report and timings.

        Returns:
            The fully decoded body and the content encoding used.
        """
        limits = self.policy.limits
        loop = asyncio.get_running_loop()
        encoding = self._content_encoding(response)
        guard.report.content_encoding = encoding
        decoder = _BodyDecoder(encoding)
        body = bytearray()
        wire_bytes = 0
        async for chunk in response.aiter_raw(limits.read_chunk_size):
            if loop.time() > guard.deadline:
                raise DeadlineExceeded(
                    f"Remote fetch exceeded the {limits.deadline_seconds:g} s "
                    "deadline while reading the body"
                )
            wire_bytes += len(chunk)
            guard.report.body_bytes = wire_bytes
            if wire_bytes > limits.max_body_bytes:
                raise TransferLimitExceeded(
                    f"Response body exceeds the {limits.max_body_bytes} byte limit"
                )
            body.extend(decoder.feed(chunk))
            guard.report.decoded_bytes = len(body)
            if len(body) > limits.max_decoded_bytes:
                raise TransferLimitExceeded(
                    "Decompressed response exceeds the "
                    f"{limits.max_decoded_bytes} byte limit"
                )
            await self._pace_reading(wire_bytes, guard)
        body.extend(decoder.finish())
        if len(body) > limits.max_decoded_bytes:
            raise TransferLimitExceeded(
                f"Decompressed response exceeds the {limits.max_decoded_bytes} "
                "byte limit"
            )
        guard.report.body_bytes = wire_bytes
        guard.report.decoded_bytes = len(body)
        return bytes(body), encoding

    def _check_headers(self, response: httpx.Response, guard: _FetchGuard) -> None:
        """Account for and enforce the response header byte limit.

        Args:
            response: The response whose headers are checked.
            guard: The per-fetch state holding the report.
        """
        header_size = self._header_size(response)
        guard.report.header_bytes += header_size
        if header_size > self.policy.limits.max_header_bytes:
            raise TransferLimitExceeded(
                "Response headers exceed the "
                f"{self.policy.limits.max_header_bytes} byte limit"
            )

    async def _send_hop_request(
        self,
        client: httpx.AsyncClient,
        url: httpx.URL,
        hop: HopReport,
        guard: _FetchGuard,
    ) -> httpx.Response:
        """Validate and send one hop's request, retrying after approval.

        Args:
            client: The client to send with.
            url: The URL being requested.
            hop: The hop report being filled in.
            guard: The per-fetch state holding the deadline.

        Returns:
            The streaming response.
        """
        request = client.build_request("GET", url)
        loop = asyncio.get_running_loop()
        while True:
            try:
                # check_target can block (e.g. an IP-literal in a restricted
                # range) before any socket is opened; the backend blocks
                # again at connect time for host names. Either way the same
                # explicit-trust retry applies.
                hop.host = self.policy.check_target(url)
                remaining = guard.deadline - loop.time()
                if remaining <= 0:
                    raise DeadlineExceeded(
                        f"Remote fetch exceeded the "
                        f"{self.policy.limits.deadline_seconds:g} s deadline"
                    )
                # Bound waiting for response headers by the overall
                # deadline, not just the read timeout.
                return await asyncio.wait_for(
                    client.send(
                        request,
                        stream=True,
                        follow_redirects=False,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as error:
                raise DeadlineExceeded(
                    f"Remote fetch exceeded the "
                    f"{self.policy.limits.deadline_seconds:g} s deadline"
                ) from error
            except HostBlocked as error:
                # Make sure the blocked addresses show up in the provenance
                # even if approval is refused and the fetch aborts here.
                guard.record_block(error)
                hop.host = error.host
                if not await self._prompt_for_trust(error):
                    raise
                # Retry the same hop now that the host has been explicitly
                # exempted (and, by contract, persisted by the caller).
                hop.warnings.append(
                    f"host '{error.host}' was blocked and approved by the "
                    "user during this fetch"
                )

    def _next_hop_url(  # pylint:disable=too-many-arguments,too-many-positional-arguments
        self,
        response: httpx.Response,
        hop: HopReport,
        current: httpx.URL,
        hop_number: int,
        guard: _FetchGuard,
    ) -> Optional[httpx.URL]:
        """Record and return the next hop URL for a redirect response.

        Args:
            response: The response to inspect.
            hop: The current hop report.
            current: The URL of the current hop.
            hop_number: The zero-based number of the current hop.
            guard: The per-fetch state holding the report.

        Returns:
            The next URL, or ``None`` if the response is not a redirect.
        """
        if response.status_code not in _REDIRECT_STATUSES:
            return None
        if hop_number == self.policy.limits.max_redirects:
            raise RedirectLimitExceeded(
                f"More than {self.policy.limits.max_redirects} redirects "
                f"while fetching {guard.report.requested_url}"
            )
        location_header = response.headers.get("location")
        if not location_header:
            raise PolicyViolation(
                f"Redirect status {response.status_code} without a Location header"
            )
        target = response.url.join(httpx.URL(location_header))
        hop.redirect_to = str(target)
        # Record a downgrade; the next iteration validates the new hop's
        # scheme, host and resolved addresses again.
        if current.scheme.lower() == "https" and target.scheme == "http":
            hop.warnings.append("redirect downgrades HTTPS to plain HTTP")
        return target

    def _build_result(
        self,
        guard: _FetchGuard,
        response: httpx.Response,
        body: bytes,
    ) -> FetchResult:
        """Decode the body and finalise the provenance report.

        Args:
            guard: The per-fetch state holding the report and timings.
            response: The final response.
            body: The decoded response body.

        Returns:
            The assembled fetch result.
        """
        content_type, raw_charset = self._parse_content_type(response)
        text, charset = self._decode_text(body, raw_charset)
        report = guard.report
        report.final_url = str(response.url)
        report.content_type = content_type
        report.charset = charset
        report.duration = asyncio.get_running_loop().time() - guard.started
        if report.duration > 0:
            report.rate_bytes_per_sec = int(report.body_bytes / report.duration)
        return FetchResult(
            text=text,
            content_type=content_type,
            final_url=response.url,
            status_code=response.status_code,
            report=report,
        )

    def _prepare(
        self, requested_url: str
    ) -> tuple[_FetchGuard, httpx.AsyncHTTPTransport, httpx.Timeout]:
        """Create the per-fetch guard, guarded transport and timeouts.

        Args:
            requested_url: The URL being fetched, for the report.

        Returns:
            The guard, transport and timeout configuration.
        """
        limits = self.policy.limits
        guard = _FetchGuard(self.policy)
        guard.report = FetchReport(requested_url=requested_url)
        transport = self._guarded_transport(guard)
        timeouts = httpx.Timeout(
            connect=limits.connect_timeout,
            read=limits.read_timeout,
            write=limits.write_timeout,
            pool=limits.connect_timeout,
        )
        return guard, transport, timeouts

    async def fetch(self, location: httpx.URL | str) -> FetchResult:
        """Fetch a document, enforcing the policy on every connection.

        Args:
            location: The URL to fetch.

        Returns:
            The decoded document together with its provenance report.

        Raises:
            PolicyViolation: For any decision that blocks the fetch.
            RemoteFetchError: For transport and resolution failures.
        """
        initial_url = (
            location if isinstance(location, httpx.URL) else httpx.URL(str(location))
        )
        guard, transport, timeouts = self._prepare(str(initial_url))
        loop = asyncio.get_running_loop()
        guard.started = loop.time()
        guard.deadline = guard.started + self.policy.limits.deadline_seconds
        try:
            async with httpx.AsyncClient(
                transport=transport,
                trust_env=False,
                timeout=timeouts,
                follow_redirects=False,
                headers={"user-agent": self.user_agent},
            ) as client:
                current = initial_url
                for hop_number in range(self.policy.limits.max_redirects + 1):
                    if loop.time() > guard.deadline:
                        raise DeadlineExceeded(
                            f"Remote fetch exceeded the "
                            f"{self.policy.limits.deadline_seconds:g} s deadline"
                        )
                    # Record the hop up front; the raw authority is only
                    # provisional -- check_target() performs the real
                    # validation (including scheme) and replaces the host.
                    hop = guard.begin_hop(
                        hop_number,
                        "GET",
                        current,
                        current.raw_host.decode("ascii", "replace")
                        if current.raw_host
                        else "",
                    )
                    response: Optional[httpx.Response] = None
                    try:
                        response = await self._send_hop_request(
                            client, current, hop, guard
                        )
                        hop.status_code = response.status_code
                        hop.reason_phrase = response.reason_phrase
                        self._check_headers(response, guard)
                        target = self._next_hop_url(
                            response, hop, current, hop_number, guard
                        )
                        if target is not None:
                            await response.aclose()
                            response = None
                            current = target
                            continue
                        body, _ = await self._read_body(response, guard)
                        return self._build_result(guard, response, body)
                    finally:
                        if response is not None:
                            await response.aclose()
        except RemoteFetchError as error:
            # Attach whatever provenance was gathered so callers can show it.
            if error.report is None:
                error.report = guard.report
            raise
        # The hop loop either returns a result or raises a RemoteFetchError;
        # this is only here to make the control flow explicit to type checkers.
        raise AssertionError("guarded fetch ended without a result")
