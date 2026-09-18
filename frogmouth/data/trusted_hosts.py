"""Explicit, persisted exceptions to the remote fetch policy.

A trusted host is a host that the *user* has explicitly approved for
contact even though :class:`~frogmouth.security.RemoteFetchPolicy` would
otherwise block it because it resolves to a restricted network. Trust is
never granted implicitly or silently: entries are only added as the result
of a user confirming a prompt, and the set is durably stored as JSON in
the application data directory.
"""

from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from typing import Final

from ..security.remote_fetch import normalize_host
from .data_directory import data_directory

TRUSTED_HOSTS_FILE: Final[str] = "trusted_hosts.json"
"""The name of the file that holds the persisted trusted hosts."""


def trusted_hosts_file() -> Path:
    """Get the location of the trusted hosts file.

    Returns:
        The path to the trusted hosts file.
    """
    return data_directory() / TRUSTED_HOSTS_FILE


def load_trusted_hosts() -> list[str]:
    """Load the persisted trusted hosts.

    Returns:
        A sorted list of normalised trusted hosts.
    """
    source = trusted_hosts_file()
    if not source.exists():
        return []
    try:
        raw = loads(source.read_text())
    except ValueError:
        return []
    if not isinstance(raw, list):
        return []
    hosts = {
        normalize_host(item) for item in raw if isinstance(item, str) and item.strip()
    }
    return sorted(hosts)


def save_trusted_hosts(hosts: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    """Persist the given set of trusted hosts.

    Args:
        hosts: The trusted hosts to persist; they are normalised and
            de-duplicated.

    Returns:
        The sorted list that was saved.
    """
    normalised = sorted({normalize_host(host) for host in hosts if host.strip()})
    trusted_hosts_file().write_text(dumps(normalised, indent=4))
    return normalised


def is_trusted_host(host: str) -> bool:
    """Check whether a host has been explicitly persisted as trusted.

    Args:
        host: The host to check (it will be normalised).

    Returns:
        ``True`` if the host is among the persisted trusted hosts.
    """
    try:
        normalised = normalize_host(host)
    except Exception:  # pylint:disable=broad-except
        return False
    return normalised in set(load_trusted_hosts())


def add_trusted_host(host: str) -> list[str]:
    """Explicitly persist a new trusted host.

    Args:
        host: The host to trust.

    Returns:
        The updated sorted list of trusted hosts.
    """
    hosts = set(load_trusted_hosts())
    hosts.add(normalize_host(host))
    return save_trusted_hosts(hosts)


def remove_trusted_host(host: str) -> list[str]:
    """Remove a persisted trusted host.

    Args:
        host: The host to revoke.

    Returns:
        The updated sorted list of trusted hosts.
    """
    hosts = set(load_trusted_hosts())
    hosts.discard(normalize_host(host))
    return save_trusted_hosts(hosts)
