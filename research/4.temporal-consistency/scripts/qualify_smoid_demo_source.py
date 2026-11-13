"""Qualify pinned official demo text and one anonymous HEAD; never read video pixels."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request,urlopen

from causal_reference_io_r75 import ROOT,BASE,write,digest,stamp

COMMIT='53e81c4eb8862d4d706da43aa007fa16f3f09610'
BLOBS={
    'README.md':'6e0639550b4935506034a49200944d47b7b52541',
    'download_dataset.py':'0c3a0005e45c6eb54592800d21748aa2fa4d205e',
    'customized_list':'53f8575ea84863d441217d044473e1887c01e413',
    'test.py':'5e05636a7bad9ddb4db728a60581c1856f346912',
    'config.py':'6787e7728e4cab36236b76993c4c6e4944756bf2',
    'train.py':'0e0d0183c38f6a938d559ff582f1227858c14adc',
}


def main():
    subprocess.run(['mountpoint','-q','/data'],check=True)
    cache=ROOT/'.cache/datasets/SMOID/source-qualification001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out=BASE/'process/external-smoid/qualification001'
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',dict(at=stamp(),commit=COMMIT,source_sha256=digest(__file__),
        protocol_sha256=digest(BASE/'docs/smoid-demo-source-qualification-protocol.md'),
        max_text_bytes_per_file=65536,archive_method='HEAD',archive_body_bytes_allowed=0,
        retries=0,pixels_allowed=False,scope='Official demo availability only; no array/GT/model qualification.'))
    started=time.perf_counter();complete=False;head_requests=0;records=[];texts={}
    try:
        for name,expected in BLOBS.items():
            url=f'https://raw.githubusercontent.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark/{COMMIT}/{name}'
            with urlopen(Request(url,headers={'User-Agent':'MobileIE-Neo-research'}),timeout=30) as response:
                raw=response.read(65537)
                assert response.status==200 and len(raw)<=65536
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert blob==expected,(name,blob,expected)
            path=cache/name;path.write_bytes(raw);texts[name]=raw.decode('utf-8')
            records.append(dict(name=name,url=url,git_blob_sha1=blob,sha256=digest(path),
                bytes=len(raw),owned_path=str(path.relative_to(ROOT))))
            write(out/'sources.json',dict(commit=COMMIT,files=records,complete=len(records)==len(BLOBS)))
        items=[line.split() for line in texts['customized_list'].splitlines() if line.strip()]
        assert len(items)==3 and all(len(r)==3 and r[1]==r[2] for r in items)
        write(out/'interface.json',dict(demo_list=items,all_demo_input_reference_paths_alias=True,
            independent_GT_confirmed=False,author_test_frame_window=32,
            author_test_histogram_scope='Full input patch passed to process_video; MAX_FRAME800 segmentation possible.',
            author_test_network_input_channels=4,author_train_target_channels=3,
            paired_benchmark_ready=False,scope='Published demo interface facts, not a claim about unavailable full dataset.'))
        identifier='1qES6teQUprs-cgL-nACzFighjm_Dho0L'
        assert identifier in texts['download_dataset.py']
        url=f'https://drive.usercontent.google.com/download?id={identifier}&export=download&confirm=t'
        head_requests+=1
        try:
            with urlopen(Request(url,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=30) as response:
                row=dict(status=response.status,host=urlparse(response.url).hostname,
                    content_type=response.headers.get_content_type(),content_length=response.headers.get('Content-Length'),
                    disposition=response.headers.get('Content-Disposition'),accept_ranges=response.headers.get('Accept-Ranges'))
        except HTTPError as error:
            row=dict(status=error.code,host=urlparse(error.url).hostname,content_type=error.headers.get_content_type(),
                     content_length=error.headers.get('Content-Length'),disposition=None,accept_ranges=None)
            error.close()
        binary=row['status']==200 and row['content_type'] in ('application/zip','application/octet-stream')
        write(out/'cam1-head.json',dict(at=stamp(),identifier=identifier,method='HEAD',**row,
            binary_archive_headers_available=binary,body_bytes_read=0,retried=False,
            scope='Headers only. Success does not establish independent GT; failure stops this acquisition route.'))
        complete=True
        print(json.dumps(dict(source_files=len(records),HEAD_status=row['status'],
            binary_archive_headers_available=binary,paired_benchmark_ready=False,pixels=0)),flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,
            source_files=len(records),archive_HEAD_requests=head_requests,archive_body_bytes_read=0,
            image_or_array_decodes=0,new_CNN=0,new_SGD=0,GPU=0,
            seconds=time.perf_counter()-started,scope='Metadata qualification only; no new benchmark completed.'))


if __name__=='__main__':
    main()
