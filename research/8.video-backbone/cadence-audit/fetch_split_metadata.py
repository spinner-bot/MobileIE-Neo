"""Read only original DAVIS2016 split text from the already pinned official ZIP."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'research/4.temporal-consistency/scripts'))
from inspect_davis_external import RemoteZip


def main():
    assets=ROOT/'.cache/research-source-reading/cadence001'
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assets.mkdir(parents=True,exist_ok=True)
    out=BASE/'process/davis-split-read001.json'
    assert not out.exists()
    old=ROOT/'research/4.temporal-consistency/process/external-davis/official-metadata.json'
    original=json.loads(old.read_text());began=time.perf_counter()
    record=dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),completed_normally=False,
        original_metadata_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        archive_url=original['archive_url'],maximum_metadata_bytes=4*1024*1024,
        image_members_read=0,new_model_or_flow_or_optimizer_calls=0)
    remote=None
    try:
        class BoundedZip(RemoteZip):
            def read(self,count=-1):
                amount=min(self.size-self.position,count if count>=0 else self.size-self.position)
                assert self.transferred+amount<=4*1024*1024
                return super().read(count)
        remote=BoundedZip(original['archive_url'])
        assert remote.size==original['archive_bytes'] and remote.etag==original['etag']
        splits={}
        with zipfile.ZipFile(remote) as archive:
            names=archive.namelist()
            for split in ('train','val'):
                member=next(n for n in names if n.endswith('/ImageSets/2016/'+split+'.txt'))
                info=archive.getinfo(member);assert info.file_size<=4096
                value=archive.read(member)
                text=value.decode('utf-8');scenes=text.split()
                assert len(scenes)==len(set(scenes))== (30 if split=='train' else 20)
                path=assets/('DAVIS2016-'+split+'.txt');path.write_bytes(value)
                splits[split]=dict(member=member,path=str(path.relative_to(ROOT)),bytes=len(value),
                    sha256=hashlib.sha256(value).hexdigest(),scenes=sorted(scenes))
        assert not set(splits['train']['scenes'])&set(splits['val']['scenes'])
        record.update(completed_normally=True,splits=splits,archive_bytes=remote.size,etag=remote.etag)
    except BaseException:
        record['failure']=traceback.format_exc();raise
    finally:
        record['seconds']=time.perf_counter()-began
        record['transferred_metadata_bytes']=remote.transferred if remote is not None else 0
        out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,transferred_metadata_bytes=remote.transferred,split_sizes={s:len(v['scenes']) for s,v in splits.items()})))


if __name__=='__main__':main()
