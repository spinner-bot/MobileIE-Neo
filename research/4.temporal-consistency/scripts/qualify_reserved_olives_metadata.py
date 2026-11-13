"""Inspect only pinned public directory JSON; reserved image bytes are never requested."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback
from urllib.request import Request,urlopen

from causal_reference_io_r75 import ROOT,BASE,write,digest,stamp

REPO='OLiVES-Lowlight/OLiVES'
REV='291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf'
RESERVED=('video_43','video_71','video_83')


def main():
    subprocess.run(['mountpoint','-q','/data'],check=True)
    out=BASE/'process/olives-reserved-metadata/qualification001'
    cache=ROOT/'.cache/datasets/OLiVES-reserved-metadata/qualification001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    inputs=['data/exploration-r4b/selection.json','data/exploration-r4b/available_directories.json',
        'data/exploration-r3/olives_manifest.json','data/exploration-r4/manifest.json',
        'data/exploration-r4b/manifest.json','data/exploration-r9-dev/manifest.json']
    write(out/'config.json',dict(at=stamp(),repo=REPO,revision=REV,reserved_ids=RESERVED,
        max_API_requests=8,max_response_bytes=2*1024*1024,retries=0,PNG_requests_allowed=0,
        protocol_sha256=digest(BASE/'docs/olives-reserved-metadata-protocol.md'),
        source_sha256=digest(__file__),input_manifests={n:digest(BASE/n) for n in inputs},
        scope='Directory metadata and existing acquired-file manifests only; no unsealing of final images.'))
    started=time.perf_counter();complete=False;requests=0;total=0;responses=[]
    try:
        saved=json.loads((BASE/inputs[0]).read_text());author={}
        for split,info in saved['author_lists'].items():
            assert hashlib.sha256(info['text'].encode()).hexdigest()==info['sha256']
            author[split]=info['text'].split()
        assert len(author['train'])==96 and len(author['test'])==41
        assert not set(author['train']) & set(author['test'])
        prior={};used_ids=set()
        for name in inputs[2:]:
            source=json.loads((BASE/name).read_text())
            for row in source['files']:
                prior.setdefault(row['sha256'],[]).append(dict(manifest=name,source_path=row['source_path']))
                if row.get('scene'):used_ids.add(row['scene'])
        assert not set(RESERVED) & used_ids

        def listing(path):
            nonlocal requests,total
            assert requests<8
            url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{path}?limit=1000'
            requests+=1
            with urlopen(Request(url,headers={'User-Agent':'MobileIE-Neo-research','Accept':'application/json'}),timeout=30) as response:
                raw=response.read(2*1024*1024+1);total+=len(raw)
                assert response.status==200 and len(raw)<=2*1024*1024
                rows=json.loads(raw);assert isinstance(rows,list)
                link=response.headers.get('Link','')
                record=dict(path=path,url=url,status=response.status,rows=len(rows),bytes=len(raw),
                    content_type=response.headers.get_content_type(),Link=link,ETag=response.headers.get('ETag'),
                    sha256=hashlib.sha256(raw).hexdigest())
            name=path.replace('/','--')+'.json';asset=cache/name;asset.write_bytes(raw)
            record['owned_path']=str(asset.relative_to(ROOT));responses.append(record)
            write(out/'responses.json',responses)
            if re.search(r'rel=["\']?next',link):
                raise RuntimeError('Next page advertised: stop incomplete catalog without guessing counts.')
            return rows

        directories={side:{Path(r['path']).name for r in listing('Frames/'+side) if r['type']=='directory'} for side in ('gt','input')}
        paired=directories['gt'] & directories['input']
        old=json.loads((BASE/inputs[1]).read_text())
        counts={split:dict(author_ids=len(names),paired_ids=[n for n in names if n in paired],
                    previously_acquired_project_ids=[n for n in names if n in used_ids],
                    paired_ids_not_in_project_acquired_manifests=[n for n in names if n in paired and n not in used_ids])
                for split,names in author.items()}
        assert counts['test']['paired_ids_not_in_project_acquired_manifests']==list(RESERVED)
        write(out/'roles.json',dict(repo=REPO,revision=REV,directories={k:sorted(v) for k,v in directories.items()},
            matches_previous_directory_sets=all(set(old[k])==v for k,v in directories.items()),
            split_counts=counts,scope='Source IDs and directory availability, not independently verified physical-scene separation.'))
        candidates=[]
        for scene in RESERVED:
            maps={};matches=[];png_counts={}
            for side in ('gt','input'):
                rows=listing(f'Frames/{side}/{scene}')
                files=[r for r in rows if r['type']=='file' and r['path'].endswith('.png')]
                assert files and all(re.fullmatch('[0-9a-f]{64}',r.get('lfs',{}).get('oid','')) for r in files)
                maps[side]={Path(r['path']).name:r for r in files};assert len(maps[side])==len(files)
                png_counts[side]=len(files)
                for r in files:
                    sha=r['lfs']['oid']
                    if sha in prior:
                        matches.append(dict(side=side,path=r['path'],sha256=sha,previously_acquired=prior[sha]))
            common=sorted(set(maps['gt']) & set(maps['input']));assert len(common)>=16
            candidate=dict(scene=scene,author_split='test',listed_PNG_counts=png_counts,
                common_frame_names=len(common),previously_acquired_exact_hash_matches=matches,
                matches_found=bool(matches),strict_unread_clip_disqualified_by_exact_match=bool(matches),
                formal_blind_test_certified=False,selected_first16_metadata=[dict(side=side,**maps[side][n]) for n in common[:16] for side in ('gt','input')],
                source_pixels_requested=0,scope='Zero matches cannot prove semantic independence or absence of all unregistered exposure.')
            candidates.append(candidate);write(out/'candidates.json',candidates)
            print(json.dumps(dict(scene=scene,PNG_counts=png_counts,exact_matches=len(matches),pixels=0)),flush=True)
        assert requests==8
        write(out/'summary.json',dict(at=stamp(),completed=True,reserved_ids=RESERVED,
            API_requests=requests,metadata_body_bytes=total,source_PNG_body_bytes=0,
            previously_acquired_unique_PNG_hashes=len(prior),
            exact_match_disqualified_ids=[r['scene'] for r in candidates if r['matches_found']],
            no_exact_match_ids=[r['scene'] for r in candidates if not r['matches_found']],
            certified_new_independent_benchmarks=0,final_pixels_unsealed=False,
            scope='Metadata-only reserved-source qualification; existing experiments/results and final pixel restrictions unchanged.'))
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,API_requests=requests,
            metadata_body_bytes=total,source_PNG_body_bytes=0,image_decodes=0,new_CNN=0,new_SGD=0,GPU=0,
            seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
