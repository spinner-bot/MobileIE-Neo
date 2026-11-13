"""Strict bounded ranges for the immutable, author-linked REDS archive."""
import http.client
import io
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_RANGE = 8 * 1024 * 1024


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.transferred = 0
        self.ranges, self.retries = [], []

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        assert 0 <= self.position <= self.size
        return self.position

    def tell(self):
        return self.position

    def range_bytes(self, start, count):
        assert 0 <= start < self.size and 0 < count <= min(MAX_RANGE, self.size - start)
        end = start + count - 1
        for attempt in range(1, 4):
            try:
                # Fresh public redirect avoids stale expiring CDN signatures. The revision and object stay fixed.
                request = Request(self.url + '?download=true&request=' + str(time.time_ns()),
                                  headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urlopen(request, timeout=45) as response:
                    assert response.status == 206, f'Range rejected ({response.status}); stop without full download'
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{self.size}'
                    length = response.headers.get('Content-Length')
                    assert length is None or int(length) == count
                    value = response.read(count + 1)
                assert len(value) == count
                self.transferred += count
                self.ranges.append(dict(start=start, bytes=count, attempt=attempt))
                return value
            except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
                self.retries.append(dict(start=start, bytes=count, attempt=attempt,
                                         error_type=type(error).__name__, status=getattr(error, 'code', None)))
                if isinstance(error, HTTPError) and error.code < 500:
                    raise
                if attempt == 3:
                    raise
                time.sleep(attempt)

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size - self.position)
        assert count <= 4 * 1024 * 1024, 'Metadata read exceeds registered bound'
        if count == 0:
            return b''
        value = self.range_bytes(self.position, count)
        self.position += count
        return value
