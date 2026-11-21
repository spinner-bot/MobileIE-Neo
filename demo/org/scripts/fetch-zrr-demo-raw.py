#!/usr/bin/env python3
"""Acquire three real Huawei RAW patches via ranges from the author's ZRR ZIP.

Reads a small prefix of the central directory and only selected members; never
downloads the 22 GB dataset. All ranges and member checksums are retained.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zlib

URL='https://download.ai-benchmark.com/public.php/dav/files/GsfikXymMKHxCwr'


def get_range(start,end):
    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'MobileIE-academic-demo/0.2'})
    with urllib.request.urlopen(request,timeout=90) as response:
        assert response.status==206,'Server must honor bounded byte ranges'
        data=response.read(end-start+2);assert len(data)==end-start+1
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        return data


def entries(data):
    offset=0
    while offset+46<=len(data):
        values=struct.unpack_from('<4s6H3L5H2L',data,offset)
        assert values[0]==b'PK\x01\x02';length=46+values[10]+values[11]+values[12]
        if offset+length>len(data):break
        name=data[offset+46:offset+46+values[10]].decode('utf-8' if values[3]&2048 else 'cp437')
        extra=data[offset+46+values[10]:offset+46+values[10]+values[11]]
        size,compressed,local=values[9],values[8],values[16]
        pos=0
        while pos+4<=len(extra):
            tag,n=struct.unpack_from('<HH',extra,pos);payload=extra[pos+4:pos+4+n];pos+=4+n
            if tag==1:
                numbers=iter(struct.unpack('<'+'Q'*(len(payload)//8),payload[:len(payload)//8*8]))
                if size==0xffffffff:size=next(numbers)
                if compressed==0xffffffff:compressed=next(numbers)
                if local==0xffffffff:local=next(numbers)
        yield {'name':name,'crc32':values[7],'compressed':compressed,'size':size,'offset':local,'method':values[4]}
        offset+=length


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    # Fixed archive size and the ZIP64 record obtained from the author endpoint.
    total=23808958387;tail=get_range(total-131072,total-1);(a.output/'tail.bin').write_bytes(tail)
    marker=tail.rfind(b'PK\x06\x06');assert marker>=0
    record=struct.unpack_from('<4sQ2H2L4Q',tail,marker);directory_size,directory_offset=record[-2:]
    data=b'';selected=[]
    for index in range(12):
        start=directory_offset+len(data);end=min(directory_offset+directory_size-1,start+262143)
        data+=get_range(start,end)
        selected=[e for e in entries(data) if e['name'].startswith('test/huawei_raw/') and e['name'].endswith('.png')][:3]
        if len(selected)==3:break
    assert len(selected)==3,'Test RAW entries not found in bounded directory prefix'
    (a.output/'central-prefix.bin').write_bytes(data)
    def download(entry):
        start=entry['offset'];header=get_range(start,start+29);v=struct.unpack('<4s5H3L2H',header);assert v[0]==b'PK\x03\x04'
        offset=start+30+v[-2]+v[-1];packed=get_range(offset,offset+entry['compressed']-1)
        image=zlib.decompress(packed,-15) if entry['method']==8 else packed
        assert len(image)==entry['size'] and (zlib.crc32(image)&0xffffffff)==entry['crc32']
        target=a.output/Path(entry['name']).name;target.write_bytes(image)
        return dict(entry,path=str(target),sha256=hashlib.sha256(image).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(download,selected))
    (a.output/'source-manifest.json').write_text(json.dumps({'author_page':'https://aiff22.github.io/pynet.html','url':URL,'archive_bytes':total,'central_directory_prefix_bytes':len(data),'members':records},indent=2)+'\n');print(json.dumps(records,indent=2))


if __name__=='__main__':main()
