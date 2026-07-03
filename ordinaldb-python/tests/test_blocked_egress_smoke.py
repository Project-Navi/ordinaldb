import importlib.util
from pathlib import Path
import socket
import sys
import tempfile
import unittest


def _load_smoke_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "examples" / "python_adapters" / "blocked_egress_smoke.py"
    spec = importlib.util.spec_from_file_location("blocked_egress_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load blocked-egress smoke module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BlockedEgressSmokeTests(unittest.TestCase):
    def test_swallowed_blocked_attempt_fails_after_script(self):
        smoke = _load_smoke_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swallow_network.py"
            path.write_text(
                "\n".join(
                    [
                        "import socket",
                        "try:",
                        "    socket.getaddrinfo('example.invalid', 443)",
                        "except OSError:",
                        "    pass",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                smoke.BlockedNetworkAssertionError,
                "socket.getaddrinfo.*example.invalid",
            ):
                smoke.run_path_with_blocked_egress(path, "swallow_network.py")

    def test_blocker_records_dns_and_socket_entry_points(self):
        smoke = _load_smoke_module()
        original_getaddrinfo = socket.getaddrinfo
        original_gethostbyaddr = socket.gethostbyaddr

        with smoke.block_socket_egress() as attempts:
            calls = (
                (
                    "socket.getaddrinfo",
                    lambda: socket.getaddrinfo("example.invalid", 443),
                ),
                (
                    "socket.gethostbyname",
                    lambda: socket.gethostbyname("example.invalid"),
                ),
                (
                    "socket.gethostbyname_ex",
                    lambda: socket.gethostbyname_ex("example.invalid"),
                ),
                ("socket.gethostbyaddr", lambda: socket.gethostbyaddr("127.0.0.1")),
                (
                    "socket.getnameinfo",
                    lambda: socket.getnameinfo(("127.0.0.1", 443), 0),
                ),
                ("socket.getfqdn", lambda: socket.getfqdn("example.invalid")),
                (
                    "socket.create_connection",
                    lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.01),
                ),
            )
            for api, call in calls:
                with self.subTest(api=api):
                    with self.assertRaises(smoke.BlockedNetworkError):
                        call()

            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                with self.assertRaises(smoke.BlockedNetworkError):
                    udp.sendto(b"x", ("127.0.0.1", 9))
            finally:
                udp.close()

        self.assertIs(socket.getaddrinfo, original_getaddrinfo)
        self.assertIs(socket.gethostbyaddr, original_gethostbyaddr)
        self.assertEqual(
            {attempt.api for attempt in attempts},
            {
                "socket.getaddrinfo",
                "socket.gethostbyname",
                "socket.gethostbyname_ex",
                "socket.gethostbyaddr",
                "socket.getnameinfo",
                "socket.getfqdn",
                "socket.create_connection",
                "socket.socket.sendto",
            },
        )


if __name__ == "__main__":
    unittest.main()
