"""Only public archive directories for the previously selected SDSD train cohort."""
from datetime import datetime
import sys
import time
import traceback

from sdsd_archive_r67 import BASE, ROOT, LISTINGS, SplitArchive, digest, volumes_for, write

sys.path.insert(0, str(ROOT/'scripts'))
from project_assets import mounted_data


def main():
    mounted_data()
    out = BASE/'process/exploration-r67/catalog001'; out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/datasets/SDSD/catalog-r67'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), source_sha256={
        name: digest(BASE/'scripts'/name) for name in ['catalog_sdsd_r67.py', 'sdsd_archive_r67.py']},
        protocol_sha256=digest(BASE/'docs/exploration-r67-sdsd-fixed-acquisition-protocol.md'),
        source_listings={str((LISTINGS/(domain+'_png-listing.json')).relative_to(ROOT)):
            digest(LISTINGS/(domain+'_png-listing.json')) for domain in ['indoor', 'outdoor']},
        image_members_fetched=0, image_decodes=0, enhancement_CNN=0, GPU=False))
    start, complete, records = time.perf_counter(), False, []
    try:
        for domain in ['indoor', 'outdoor']:
            volumes = volumes_for(domain); remote = SplitArchive(volumes)
            rows, tail, central, eocd = remote.directory()
            directory = cache/(domain+'.index.json'); write(directory, rows)
            (cache/(domain+'.tail.bin')).write_bytes(tail); (cache/(domain+'.central.bin')).write_bytes(central)
            record = dict(at=datetime.now().astimezone().isoformat(), domain=domain, volumes=volumes, EOCD=eocd,
                index=str(directory.relative_to(ROOT)), index_sha256=digest(directory),
                tail_sha256=digest(cache/(domain+'.tail.bin')), central_sha256=digest(cache/(domain+'.central.bin')),
                transferred_directory_bytes=remote.transferred, technical_retries=remote.retries,
                observed_etags=remote.observed_etags, image_members_fetched=0, images_decoded=0,
                scope='Directory metadata only; no test/train image content decoded, no quality-based selection.')
            write(out/(domain+'.json'), record); records.append(record)
            print({'domain':domain, 'members':len(rows), 'first_names':[r['name'] for r in rows[:4]],
                   'directory_bytes':remote.transferred}, flush=True)
        write(out/'catalog.json', dict(complete=True, domains=records)); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, domains_completed=len(records),
              image_members_fetched=0, image_decodes=0, enhancement_CNN=0, GPU=False,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
