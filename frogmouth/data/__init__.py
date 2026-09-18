"""Provides tools for saving and loading application data."""

from .bookmarks import Bookmark, load_bookmarks, save_bookmarks
from .config import Config, load_config, save_config
from .history import load_history, save_history
from .trusted_hosts import (
    add_trusted_host,
    is_trusted_host,
    load_trusted_hosts,
    remove_trusted_host,
    save_trusted_hosts,
)

__all__ = [
    "Bookmark",
    "Config",
    "add_trusted_host",
    "is_trusted_host",
    "load_bookmarks",
    "load_config",
    "load_history",
    "load_trusted_hosts",
    "remove_trusted_host",
    "save_bookmarks",
    "save_config",
    "save_history",
    "save_trusted_hosts",
]
