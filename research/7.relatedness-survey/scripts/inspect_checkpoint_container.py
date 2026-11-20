"""Inspect a retained checkpoint ZIP and pickle opcodes; never deserialize tensors."""
import argparse
import collections
import datetime
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

B = Path(__file__).resolve().parents[1]


def inspect(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    assert receipt['complete'] is True and receipt['status'] == 'success'
    source = B / receipt['archive_path']
    assert source.stat().st_size == receipt['bytes']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['sha256']
    if receipt.get('declared_bytes') is not None:
        assert receipt['bytes'] == receipt['declared_bytes']
    with zipfile.ZipFile(source) as z:
        members = z.infolist()
        assert sum(m.file_size for m in members) <= 256 * 1024 * 1024
        assert z.testzip() is None, 'CRC failure'
        metadata = [m for m in members if m.filename.endswith('/data.pkl')]
        assert len(metadata) == 1 and metadata[0].file_size < 4 * 1024 * 1024
        data = z.read(metadata[0])
        operations = collections.Counter()
        globals_seen = set()
        strings = []
        for op, arg, pos in pickletools.genops(data):
            operations[op.name] += 1
            if op.name == 'GLOBAL':
                globals_seen.add(arg)
            if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'):
                strings.append(arg)
        model_keys = [s for s in strings if any(t in s for t in
            ('TSM.', 'linear1_experts.', 'linear2_experts.', 'relative_position_bias_table'))]
        task_literals = [s for s in strings if s.lower() in
            ('denoise', 'dehaze', 'derain', 'deblur', 'enhance', 'blind', 'base_prompts', 'candidates')]
        storage = [m for m in members if '/data/' in m.filename]
        return dict(receipt=str(receipt_path.relative_to(B)), archive=receipt['archive_path'],
            sha256=receipt['sha256'], bytes=receipt['bytes'], declared_bytes=receipt.get('declared_bytes'),
            zip_members=len(members), all_member_CRC_passed=True,
            total_uncompressed_bytes=sum(m.file_size for m in members),
            storage_members=len(storage), storage_bytes=sum(m.file_size for m in storage),
            metadata_member=metadata[0].filename, metadata_bytes=len(data),
            opcode_counts=dict(operations), global_references=sorted(globals_seen),
            model_key_strings_count=len(model_keys), model_key_examples=model_keys[:24],
            literal_task_mapping_strings=task_literals,
            interpretation='Structural ZIP/CRC and symbolic pickle opcode inspection only. GLOBAL/REDUCE opcodes recorded but never executed; no pickle.load/torch.load, imports from checkpoint, tensor allocation, or model run. Key strings do not establish full shapes, correctness, mapping semantics, or paper result reproduction.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipts', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = B / a.output
    assert output.resolve().is_relative_to(B.resolve())
    assert not output.exists(), 'Keep prior inspections'
    rows = [inspect(B / path) for path in a.receipts]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        checked_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        method='stdlib zipfile/hashlib/pickletools, no deserialization or upstream execution',
        rows=rows), ensure_ascii=False, indent=2) + '\n')
    print('Inspected', len(rows), 'ZIPs without deserializing; output', a.output)


if __name__ == '__main__':
    main()
