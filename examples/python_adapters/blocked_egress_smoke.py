"""Run adapter edge examples with Python socket egress blocked.

This catches accidental socket use in the adapter examples. It is not a
replacement for OS, container, or firewall egress controls in strict deployments.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
SMOKES = (
    "langchain_edge_smoke.py",
    "llama_index_edge_smoke.py",
    "haystack_edge_smoke.py",
    "agno_edge_smoke.py",
)


@dataclass(frozen=True)
class BlockedNetworkAttempt:
    api: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class BlockedNetworkError(OSError):
    def __init__(self, attempt: BlockedNetworkAttempt) -> None:
        super().__init__(
            f"network egress blocked during adapter smoke: {_format_attempt(attempt)}"
        )
        self.attempt = attempt


class BlockedNetworkAssertionError(AssertionError):
    pass


def _format_attempt(attempt: BlockedNetworkAttempt) -> str:
    return f"{attempt.api} args={attempt.args!r} kwargs={attempt.kwargs!r}"


def _format_attempts(attempts: list[BlockedNetworkAttempt]) -> str:
    shown = "; ".join(_format_attempt(attempt) for attempt in attempts[:5])
    if len(attempts) > 5:
        shown = f"{shown}; ... {len(attempts) - 5} more"
    return shown


def _record_blocked(
    attempts: list[BlockedNetworkAttempt],
    api: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    attempt = BlockedNetworkAttempt(api=api, args=args, kwargs=dict(kwargs))
    attempts.append(attempt)
    raise BlockedNetworkError(attempt)


def _make_blocker(attempts: list[BlockedNetworkAttempt], api: str):
    def blocked(*args: Any, **kwargs: Any) -> None:
        _record_blocked(attempts, api, *args, **kwargs)

    return blocked


@contextlib.contextmanager
def block_socket_egress() -> Iterator[list[BlockedNetworkAttempt]]:
    attempts: list[BlockedNetworkAttempt] = []
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex
    original_gethostbyaddr = socket.gethostbyaddr
    original_getnameinfo = socket.getnameinfo
    original_getfqdn = socket.getfqdn

    class BlockedSocket(original_socket):
        def connect(self, address):
            _record_blocked(attempts, "socket.socket.connect", address)

        def connect_ex(self, address):
            _record_blocked(attempts, "socket.socket.connect_ex", address)

        def sendto(self, *args, **kwargs):
            _record_blocked(attempts, "socket.socket.sendto", *args, **kwargs)

    socket.socket = BlockedSocket
    socket.create_connection = _make_blocker(attempts, "socket.create_connection")
    socket.getaddrinfo = _make_blocker(attempts, "socket.getaddrinfo")
    socket.gethostbyname = _make_blocker(attempts, "socket.gethostbyname")
    socket.gethostbyname_ex = _make_blocker(attempts, "socket.gethostbyname_ex")
    socket.gethostbyaddr = _make_blocker(attempts, "socket.gethostbyaddr")
    socket.getnameinfo = _make_blocker(attempts, "socket.getnameinfo")
    socket.getfqdn = _make_blocker(attempts, "socket.getfqdn")
    try:
        yield attempts
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyname = original_gethostbyname
        socket.gethostbyname_ex = original_gethostbyname_ex
        socket.gethostbyaddr = original_gethostbyaddr
        socket.getnameinfo = original_getnameinfo
        socket.getfqdn = original_getfqdn


def assert_no_blocked_network_attempts(
    attempts: list[BlockedNetworkAttempt],
    label: str,
) -> None:
    if attempts:
        raise BlockedNetworkAssertionError(
            f"blocked network egress was attempted during {label}: {_format_attempts(attempts)}"
        )


def run_path_with_blocked_egress(path: str | os.PathLike[str], label: str | None = None) -> None:
    smoke_path = Path(path)
    smoke_label = label or str(smoke_path)
    with block_socket_egress() as attempts:
        try:
            runpy.run_path(str(smoke_path), run_name="__main__")
        except BaseException as exc:
            if attempts:
                raise BlockedNetworkAssertionError(
                    f"blocked network egress was attempted during {smoke_label}: "
                    f"{_format_attempts(attempts)}"
                ) from exc
            raise
    assert_no_blocked_network_attempts(attempts, smoke_label)


def _set_offline_env() -> None:
    os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
    os.environ.setdefault("HAYSTACK_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("POSTHOG_DISABLED", "1")
    os.environ.setdefault("AGNO_TELEMETRY", "false")


def _run_one(script: str) -> None:
    if script not in SMOKES:
        raise SystemExit(f"unknown adapter smoke script: {script}")
    path = ROOT / "examples" / "python_adapters" / script
    run_path_with_blocked_egress(path, script)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    _set_offline_env()

    if args[:1] == ["--run-one"]:
        if len(args) != 2:
            raise SystemExit("usage: blocked_egress_smoke.py --run-one <smoke-script>")
        _run_one(args[1])
        return
    if args:
        raise SystemExit("usage: blocked_egress_smoke.py [--run-one <smoke-script>]")

    for script in SMOKES:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run-one", script],
            check=True,
        )


if __name__ == "__main__":
    main()
