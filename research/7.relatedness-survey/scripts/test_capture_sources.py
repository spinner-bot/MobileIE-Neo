"""Controlled HTTP-response regressions; no network, model or upstream imports."""
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import capture_sources as capture


class Response:
    status=200
    url='https://example.invalid/public-resource'

    def __init__(self, blocks, length=None, status=200):
        self.blocks=iter(blocks)
        self.headers={'Content-Type':'application/octet-stream'}
        if length is not None:self.headers['Content-Length']=str(length)
        self.status=status

    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):
        value=next(self.blocks,b'')
        if isinstance(value,Exception):raise value
        return value


class Integrity(unittest.TestCase):
    def run_response(self,response,limit=1000):
        owned=capture.ARCHIVE.parent/'controlled-http-checks'
        owned.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=owned) as folder:
            root=Path(folder)
            (root/'collection/receipts').mkdir(parents=True)
            (root/'collection/raw').mkdir()
            (root/'archive').mkdir()
            with patch.object(capture,'BASE',root),patch.object(capture,'ARCHIVE',root/'archive'),patch.object(capture.urllib.request,'urlopen',return_value=response):
                capture.capture({'label':'controlled','url':response.url,'max_bytes':limit})
            return json.loads((root/'collection/receipts/controlled.json').read_text())

    def test_multiple_short_blocks_not_truncated(self):
        r=self.run_response(Response([b'ab',b'c',b'd'],4))
        self.assertEqual((r['status'],r['bytes'],r['complete']),('success',4,True))

    def test_short_eof_rejected(self):
        r=self.run_response(Response([b'ab'],4))
        self.assertEqual((r['status'],r['bytes'],r['complete']),('length_mismatch_incomplete',2,False))

    def test_size_cap_preserved(self):
        r=self.run_response(Response([b'abcdef'],6),3)
        self.assertEqual((r['status'],r['bytes'],r['complete']),('size_limit_incomplete',3,False))

    def test_empty_202_not_resource_success(self):
        r=self.run_response(Response([],0,202))
        self.assertEqual((r['status'],r['complete']),('empty_response',False))

    def test_partial_exception_retains_prefix(self):
        r=self.run_response(Response([b'ab',http.client.IncompleteRead(b'c',2)],5))
        self.assertEqual((r['status'],r['bytes'],r['complete']),('length_mismatch_incomplete',3,False))

    def test_zip_signature_is_not_complete_zip(self):
        r=self.run_response(Response([b'PK\x03\x04garbage'],11))
        self.assertEqual((r['status'],r['complete']),('invalid_zip_container',False))

    def test_complete_zip_accepted(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as archive:archive.writestr('source.txt','static resource')
        raw=buf.getvalue()
        r=self.run_response(Response([raw[:10],raw[10:]],len(raw)))
        self.assertEqual((r['status'],r['complete']),('success',True))


if __name__=='__main__':unittest.main()
