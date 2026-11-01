"""Every HTTP call the helper scripts make, against a hostile local server.

A local port is not automatically a trusted one: while ollama.service is
stopped, any process on this machine can bind 127.0.0.1:11434 and answer. The
byte cap on the final response (MAX_API_BYTES) was never the only read, though.
urllib's default HTTPRedirectHandler drains a 3xx body with an unbounded
`fp.read()` before it follows the redirect, so an endless redirect body was
read in full -- the socket timeout bounds each recv, not the total, and a
server that keeps sending never trips it. Raised by the marketplace security
review against v0.5.0 (`d9b94f6`).

Each server here counts the bytes it managed to hand to the kernel before the
client hung up. That count includes whatever the loopback socket buffers
absorbed without the client reading anything, so the bound asserted is loose
(16 MiB) against a body four times that size: a client that drains the body
reaches 64 MiB, one that refuses it stays at a few.
"""

import contextlib
import http.server
import io
import os
import socket
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import colophon_action as action
import colophon_collect as collect

HOSTILE_BODY_BYTES = 64 << 20
SENT_BOUND_BYTES = 16 << 20
CHUNK = 64 << 10
GOOD_JSON = b'{"version":"0.0.0"}'


class Hostile(object):
    """A loopback server whose response to everything except /ok is `mode`.

    Modes: "redirect" (302, endless body), "error" (400, endless body),
    "big" (200, endless body), "drip" (200, one byte per 0.1s for 6s).
    /ok always answers a small valid JSON object, so a client that follows a
    redirect there gets a plausible success -- which is what makes following
    one observable.
    """

    def __init__(self, test, mode):
        self.mode = mode
        self.sent = 0
        self.done = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                owner.respond(self)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                owner.respond(self)

            def log_message(self, *args):
                pass  # keep test output pristine

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        test.addCleanup(thread.join, 5)
        test.addCleanup(self.server.server_close)
        test.addCleanup(self.server.shutdown)
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])

    def respond(self, handler):
        handler.close_connection = True
        if handler.path == "/ok":
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(GOOD_JSON)))
            handler.end_headers()
            handler.wfile.write(GOOD_JSON)
            return
        try:
            if self.mode == "redirect":
                handler.send_response(302)
                handler.send_header("Location", self.base + "/ok")
            elif self.mode == "error":
                handler.send_response(400)
            else:
                handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            if self.mode == "drip":
                handler.send_header("Content-Length", "100000")
            handler.end_headers()
            handler.wfile.flush()
            if self.mode == "drip":
                stop = time.monotonic() + 6
                while time.monotonic() < stop:
                    handler.wfile.write(b" ")
                    handler.wfile.flush()
                    self.sent += 1
                    time.sleep(0.1)
                return
            # No Content-Length: the body runs until the connection closes,
            # exactly as a hostile server would send it.
            block = b"x" * CHUNK
            while self.sent < HOSTILE_BODY_BYTES:
                handler.wfile.write(block)
                self.sent += len(block)
            handler.wfile.flush()
        except OSError:
            pass  # the client hung up, which is the point
        finally:
            self.done.set()

    def sent_after_client_returns(self):
        # The handler is either finished or about to hit EPIPE/ECONNRESET on
        # its next write now that the client has closed.
        self.done.wait(10)
        return self.sent


def quietly(fn, *args):
    """Call `fn`, swallowing what post_json writes to stderr by contract."""
    with contextlib.redirect_stderr(io.StringIO()) as err:
        result = fn(*args)
    return result, err.getvalue()


class RedirectBodyIsNeverReadTest(unittest.TestCase):
    """The reviewer's finding, once per call path. Each fails on v0.5.0."""

    def test_api_get_refuses_a_redirect_without_draining_its_body(self):
        server = Hostile(self, "redirect")
        payload, latency = collect.api_get(server.base, "/api/version", 2)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)
        self.assertIsNone(payload)
        self.assertIsNone(latency)

    def test_api_reachable_refuses_a_redirect_without_draining_its_body(self):
        server = Hostile(self, "redirect")
        reachable = action.api_reachable(server.base, timeout=2)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)
        self.assertFalse(reachable)

    def test_post_json_refuses_a_redirect_without_draining_its_body(self):
        server = Hostile(self, "redirect")
        code, err = quietly(action.post_json, server.base + "/api/generate",
                            {"model": "x:1", "keep_alive": "5m"})
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)
        self.assertEqual(code, 1)
        self.assertIn("302", err)

    def test_a_redirect_is_refused_even_when_its_body_is_small(self):
        # Not following is the rule, not a size heuristic: Ollama's API never
        # redirects, so any redirect is something else answering the port.
        class SmallRedirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/ok":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(GOOD_JSON)))
                    self.end_headers()
                    self.wfile.write(GOOD_JSON)
                    return
                self.send_response(307)
                self.send_header("Location", "/ok")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), SmallRedirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_address[1])

        payload, _ = collect.api_get(base, "/api/version", 2)
        self.assertIsNone(payload)


class OversizedBodyTest(unittest.TestCase):
    def test_api_get_stops_reading_an_endless_200(self):
        server = Hostile(self, "big")
        payload, _ = collect.api_get(server.base, "/api/ps", 10)
        self.assertIsNone(payload)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)

    def test_api_reachable_treats_an_oversized_200_as_not_ollama(self):
        server = Hostile(self, "big")
        self.assertFalse(action.api_reachable(server.base, timeout=5))
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)

    def test_post_json_treats_an_oversized_200_as_a_failure(self):
        server = Hostile(self, "big")
        code, err = quietly(action.post_json, server.base + "/api/generate",
                            {"model": "x:1", "keep_alive": "5m"})
        self.assertEqual(code, 1)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)

    def test_post_json_reads_only_a_one_line_error_detail(self):
        server = Hostile(self, "error")
        code, err = quietly(action.post_json, server.base + "/api/generate",
                            {"model": "x:1", "keep_alive": "5m"})
        self.assertEqual(code, 1)
        self.assertIn("400", err)
        self.assertLessEqual(len(err), action.MAX_ERROR_DETAIL_CHARS + 200)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)

    def test_api_get_does_not_read_an_endless_error_body(self):
        server = Hostile(self, "error")
        payload, _ = collect.api_get(server.base, "/api/version", 2)
        self.assertIsNone(payload)
        self.assertLess(server.sent_after_client_returns(), SENT_BOUND_BYTES)


class WallClockDeadlineTest(unittest.TestCase):
    """A socket timeout bounds each recv, so one byte every 0.1s never trips
    it. The timeout passed in is also a deadline for the whole exchange."""

    def test_api_get_gives_up_on_a_drip_fed_body(self):
        server = Hostile(self, "drip")
        started = time.monotonic()
        payload, _ = collect.api_get(server.base, "/api/version", 1)
        elapsed = time.monotonic() - started
        self.assertIsNone(payload)
        self.assertLess(elapsed, 3)

    def test_post_json_gives_up_on_a_drip_fed_body(self):
        server = Hostile(self, "drip")
        original = action.LOAD_POST_TIMEOUT_SEC
        action.LOAD_POST_TIMEOUT_SEC = 1
        self.addCleanup(setattr, action, "LOAD_POST_TIMEOUT_SEC", original)
        started = time.monotonic()
        code, _ = quietly(action.post_json, server.base + "/api/generate",
                          {"model": "x:1", "keep_alive": "5m"})
        self.assertEqual(code, 1)
        self.assertLess(time.monotonic() - started, 3)


class StillWorksTest(unittest.TestCase):
    def test_a_normal_answer_still_reaches_every_caller(self):
        server = Hostile(self, "redirect")  # /ok is always well-behaved
        payload, latency = collect.api_get(server.base, "/ok", 2)
        self.assertEqual(payload, {"version": "0.0.0"})
        self.assertIsInstance(latency, int)
        code, err = quietly(action.post_json, server.base + "/ok", {})
        self.assertEqual(code, 0, err)

    def test_a_refused_connection_is_still_quiet(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        base = "http://127.0.0.1:" + str(port)
        self.assertEqual(collect.api_get(base, "/api/version", 1), (None, None))
        self.assertFalse(action.api_reachable(base, timeout=1))


class OneHttpPathTest(unittest.TestCase):
    def test_no_helper_script_opens_a_url_except_through_colophon_http(self):
        # A new call site that goes straight to urlopen gets urllib's default
        # redirect handler back, and with it the unbounded drain.
        scripts = os.path.join(ROOT, "scripts")
        offenders = []
        for name in sorted(os.listdir(scripts)):
            if not name.endswith(".py") or name == "colophon_http.py":
                continue
            with open(os.path.join(scripts, name)) as handle:
                text = handle.read()
            for needle in ("urlopen(", "build_opener(", "http.client.HTTP"
                           "Connection("):
                if needle in text:
                    offenders.append(name + ": " + needle)
        self.assertEqual(offenders, [])
        self.assertTrue(os.path.exists(os.path.join(scripts, "colophon_http.py")))


if __name__ == "__main__":
    unittest.main()
