"""Fetch the three preregistered original phone videos, with no image decoding."""
import hashlib
import json
from pathlib import Path
import struct
import zlib

from inspect_lliv_archive import BASE, ROOT, URL, RangeOnlyZip


def main():
    metadata_path = BASE/'process/external-phone/archive-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    index_path = ROOT/metadata['cached_index']
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == metadata['index_sha256']
    index = json.loads(index_path.read_text())
    members = index['members']
    offsets = sorted([r['offset'] for r in members]+[index['central_directory_start']])
    next_offset = dict(zip(offsets, offsets[1:]))
    selected = []
    for phone in ['Huawei Mate 20 Pro', 'Pixel 3', 'iPhone 11']:
        choices = sorted([r for r in members if Path(r['name']).parent.name == phone
                          and Path(r['name']).suffix.lower() in ['.mp4', '.mov', '.m4v']], key=lambda r: r['name'])
        selected.append({'phone': phone, **choices[0]})
    assert [r['size'] for r in selected] == [10162099, 30514807, 9556797]
    protocol = BASE/'docs/exploration-r31-protocol.md'
    out = BASE/'data/external-phone'
    out.mkdir(parents=True, exist_ok=True)
    selection = out/'selection.json'
    assert not selection.exists() and not (out/'manifest.json').exists()
    selection.write_text(json.dumps({'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'source_index_sha256': metadata['index_sha256'], 'selected': selected,
              'scope': 'Three selected original videos; no frame quality or decoding used for selection.'}, ensure_ascii=False, indent=2)+'\n')
    remote = RangeOnlyZip(URL)
    assert remote.size == metadata['archive_bytes']
    frames = []
    for row in selected:
        remote.seek(row['offset'])
        block = remote.read(next_offset[row['offset']]-row['offset'])
        assert block[:4] == b'PK\x03\x04'
        flags, method = struct.unpack_from('<HH', block, 6)
        assert not flags & 1 and method == row['method'] and method in [0, 8]
        n, extra = struct.unpack_from('<HH', block, 26)
        assert block[30:30+n].decode('utf8') == row['name']
        packed = block[30+n+extra:30+n+extra+row['compressed_size']]
        assert len(packed) == row['compressed_size']
        value = packed if method == 0 else zlib.decompress(packed, -zlib.MAX_WBITS)
        assert len(value) == row['size'] and f'{zlib.crc32(value) & 0xffffffff:08x}' == row['crc32']
        target = ROOT/'.cache/datasets/LLIV-Phone/pilot'/row['phone']/Path(row['name']).name
        assert target.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        frames.append({**row, 'path': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(value).hexdigest()})
        print(json.dumps({'phone': row['phone'], 'bytes': len(value), 'CRC_verified': True}), flush=True)
    (out/'manifest.json').write_text(json.dumps({'videos': frames, 'archive_bytes': remote.size,
                'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                'decoded_video_frames': 0, 'scope': 'Only three original video members; no GT acquired or quality results.'},
                ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
