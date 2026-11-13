"""Register then fetch official TRAIN videos excluding the three existing DEV IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from fetch_davis_pilot import BASE, ROOT, RemoteZip, fetch

PROTOCOL = BASE/'docs/exploration-r32-data-protocol.md'
METADATA = BASE/'data/exploration-r32-training/selection.json'
DESTINATION = 'DAVIS2017-train-rest'


def register():
    assert not METADATA.exists()
    source = BASE/'process/external-davis/official-metadata.json'
    old = json.loads(source.read_text())
    excluded = sorted(old['pilot_members'])
    scenes = sorted(set(old['splits']['train']['scenes'])-set(excluded))
    assert len(scenes) == 57 and set(excluded) == {'bear', 'lucia', 'walking'}
    assert not set(scenes) & set(old['splits']['val']['scenes'])
    remote = RemoteZip(old['archive_url'])
    assert remote.size == old['archive_bytes'] and remote.etag == old['etag']
    rows, counts = [], {}
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        offsets = sorted([i.header_offset for i in infos]+[archive.start_dir])
        end_offsets = dict(zip(offsets, offsets[1:]))
        for scene in scenes:
            selected = sorted((i for i in infos if '/JPEGImages/480p/'+scene+'/' in i.filename and i.filename.endswith('.jpg')),
                              key=lambda i: i.filename)[:64]
            assert len(selected) >= 2
            counts[scene] = len(selected)
            for i in selected:
                assert i.compress_type in [0, 8]
                end = end_offsets[i.header_offset]-1
                assert end-i.header_offset+1 <= i.compress_size+128*1024
                rows.append({'scene': scene, 'name': i.filename, 'size': i.file_size, 'crc32': f'{i.CRC:08x}',
                             'offset': i.header_offset, 'end': end, 'compressed_size': i.compress_size, 'method': i.compress_type})
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    value = {'archive_url': remote.url, 'archive_bytes': remote.size, 'etag': remote.etag,
             'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
             'official_metadata_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'excluded_development_videos': excluded, 'training_videos': scenes, 'frame_counts': counts,
             'frames': rows, 'selected_frames': len(rows), 'selected_source_bytes': sum(r['size'] for r in rows),
             'image_members_read': 0, 'validation_image_members_read': 0, 'training_started': False}
    METADATA.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'training_videos': len(scenes), 'selected_frames': len(rows), 'source_bytes': value['selected_source_bytes'],
                      'shorter_than64_videos': sum(n < 64 for n in counts.values()), 'image_members_read': 0}), flush=True)


def download():
    selection = json.loads(METADATA.read_text())
    assert selection['protocol_sha256'] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    out = METADATA.with_name('manifest.json')
    assert not out.exists()
    remote = RemoteZip(selection['archive_url'])
    assert remote.size == selection['archive_bytes'] and remote.etag == selection['etag']
    def one(row):
        record = {k: row[k] for k in ['name', 'size', 'crc32']}
        item = record, remote.url, remote.size, remote.etag, row['offset'], row['end'], row['compressed_size'], row['method']
        return {'scene': row['scene'], **fetch(item, destination=DESTINATION)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, selection['frames']))
    assert len(rows) == selection['selected_frames'] and len({r['scene'] for r in rows}) == 57
    value = {'selection_sha256': hashlib.sha256(METADATA.read_bytes()).hexdigest(),
             'protocol_sha256': selection['protocol_sha256'], 'frames': rows, 'videos': 57,
             'excluded_development_videos': selection['excluded_development_videos'], 'validation_image_members_read': 0,
             'decoded_image_pixels': 0, 'training_started': False,
             'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'completed': True, 'frames': len(rows), 'training_started': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    assert args.register != args.download, 'Choose exactly one stage; commit metadata before downloading'
    register() if args.register else download()


if __name__ == '__main__':
    main()
