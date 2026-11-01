"""The one HTTP path both helper scripts use to talk to the Ollama API.

A local port is not automatically a trusted one: while ollama.service is
stopped, any process on this machine can bind 127.0.0.1:11434 and answer. So
every byte read from a response is bounded here, in one place:

- **Redirects are refused, not followed.** urllib's default
  HTTPRedirectHandler drains the redirect body with an unbounded `fp.read()`
  before following, so a 302 with an endless body was read in full, ahead of
  any byte cap on the final response. Ollama's API never redirects; a redirect
  means something else is on the port. The refusal closes the response unread.
- **Bodies are read in bounded pieces** -- `max_bytes + 1` at most, so an
  oversized answer is detected rather than silently truncated -- and error
  bodies at most `detail_bytes` (zero means never read).
- **`timeout` is also a wall-clock deadline** for the body. A socket timeout
  bounds each recv, so a server dripping one byte a second never trips it;
  `read1` returns after a single recv, which lets the deadline be checked
  between them. Worst case is the deadline plus one socket timeout.

Header parsing stays bounded by http.client itself (100 header lines of at most
64 KiB each). Stdlib only; see CONTRIBUTING.md invariant 1.
"""

import http.client
import time
import urllib.error
import urllib.request

READ_CHUNK_BYTES = 64 << 10


class FetchError(Exception):
    """Any failure to get a usable answer. `code` is the HTTP status, if one
    arrived; `detail` is the start of the error body, if one was read."""

    def __init__(self, message, code=None, detail=""):
        Exception.__init__(self, message)
        self.code = code
        self.detail = detail


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def _refuse(self, req, fp, code, msg, headers):
        # Close without reading: the stock handler's fp.read() is the drain
        # this module exists to avoid.
        if fp is not None:
            fp.close()
        raise urllib.error.HTTPError(req.full_url, code,
                                     "redirect refused: " + str(msg),
                                     headers, None)

    http_error_301 = _refuse
    http_error_302 = _refuse
    http_error_303 = _refuse
    http_error_307 = _refuse
    http_error_308 = _refuse


# build_opener drops its default HTTPRedirectHandler when given a subclass.
_OPENER = urllib.request.build_opener(_RefuseRedirects)


class _DeadlineExceeded(Exception):
    pass


def _bounded_read(fp, limit, deadline):
    """Up to `limit` bytes, one recv at a time, stopping at `deadline`."""
    read = getattr(fp, "read1", None) or fp.read
    chunks = []
    total = 0
    while total < limit:
        if time.monotonic() >= deadline:
            raise _DeadlineExceeded()
        chunk = read(min(READ_CHUNK_BYTES, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def fetch(url, timeout, max_bytes, data=None, headers=None, detail_bytes=0):
    """GET (or POST, with `data`) `url`; the body, or raise FetchError."""
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        response = _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = b""
        try:
            with error:
                fp = getattr(error, "fp", None)
                if detail_bytes > 0 and fp is not None:
                    detail = _bounded_read(fp, detail_bytes, deadline)
        except (_DeadlineExceeded, http.client.HTTPException, OSError,
                ValueError):
            # Reading the error body can itself hit IncompleteRead, which is
            # an HTTPException and NOT an OSError (CONTRIBUTING.md trap #25).
            pass
        raise FetchError("returned " + str(error.code), error.code,
                         detail.decode("utf-8", "replace").strip())
    except (urllib.error.URLError, http.client.HTTPException, OSError,
            ValueError) as error:
        raise FetchError("could not reach " + url + ": " + str(error))
    try:
        with response:
            body = _bounded_read(response, max_bytes + 1, deadline)
    except _DeadlineExceeded:
        raise FetchError("timed out reading " + url)
    except (http.client.HTTPException, OSError, ValueError) as error:
        raise FetchError("could not read " + url + ": " + str(error))
    if len(body) > max_bytes:
        raise FetchError(url + " answered more than " + str(max_bytes)
                         + " bytes")
    return body
