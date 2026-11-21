"""Read bounded FlatBuffer metadata only. Never import a model/runtime or invoke inference.

Field numbers/types follow the retained official TensorFlow v2.20.0 schema.fbs.
This checks storage structure and declared tensor/operator types, not backend support,
quantization fidelity, delegation coverage, measured speed or numerical correctness.
"""
import argparse
import collections
import hashlib
import json
import re
import struct
from pathlib import Path


class Reader:
    def __init__(self, raw):
        self.raw = raw

    def scalar(self, fmt, pos):
        size = struct.calcsize('<' + fmt)
        assert 0 <= pos <= len(self.raw) - size, (fmt, pos)
        return struct.unpack_from('<' + fmt, self.raw, pos)[0]

    def field(self, table, index):
        vtable = table - self.scalar('i', table)
        length = self.scalar('H', vtable)
        assert length >= 4 and vtable + length <= len(self.raw)
        slot = vtable + 4 + 2 * index
        if slot + 2 > vtable + length:
            return None
        offset = self.scalar('H', slot)
        return table + offset if offset else None

    def value(self, table, index, fmt, default=0):
        p = self.field(table, index)
        return default if p is None else self.scalar(fmt, p)

    def pointer(self, table, index):
        p = self.field(table, index)
        return p + self.scalar('I', p) if p is not None else None

    def vector(self, table, index, fmt='i', offsets=False):
        p = self.pointer(table, index)
        if p is None:
            return []
        n = self.scalar('I', p)
        assert n <= 2000000
        size = 4 if offsets else struct.calcsize('<' + fmt)
        assert p + 4 + size * n <= len(self.raw)
        if offsets:
            return [p + 4 + 4 * k + self.scalar('I', p + 4 + 4 * k) for k in range(n)]
        return [self.scalar(fmt, p + 4 + size * k) for k in range(n)]

    def vector_length(self, table, index):
        p = self.pointer(table, index)
        return 0 if p is None else self.scalar('I', p)

    def string(self, table, index):
        p = self.pointer(table, index)
        if p is None:
            return None
        n = self.scalar('I', p)
        assert n < 65536 and p + 4 + n <= len(self.raw)
        return self.raw[p + 4:p + 4 + n].decode('utf-8', errors='replace')


def enum_values(schema, name):
    body = re.search(r'enum\s+' + re.escape(name) + r'\s*:[^{]+\{(.*?)\n\}', schema, re.S)[1]
    return {int(n): s for s, n in re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+)', body, re.M)}


def inspect(path, schema):
    raw = path.read_bytes()
    assert raw[4:8] == b'TFL3', 'Not a TFL3 FlatBuffer'
    r = Reader(raw)
    root = r.scalar('I', 0)
    assert r.value(root, 0, 'I') == 3
    types = enum_values(schema, 'TensorType')
    builtins = enum_values(schema, 'BuiltinOperator')
    codes = []
    for t in r.vector(root, 1, offsets=True):
        code = max(r.value(t, 0, 'b'), r.value(t, 3, 'i'))
        codes.append({'builtin_code': code, 'name': builtins.get(code, f'UNKNOWN_{code}'),
                      'custom': r.string(t, 1), 'version': r.value(t, 2, 'i', 1)})
    buffers = r.vector(root, 4, offsets=True)
    buffer_lengths = [r.vector_length(b, 0) for b in buffers]
    for b, n in zip(buffers, buffer_lengths):
        p = r.pointer(b, 0)
        assert not n or p is not None and p + 4 + n <= len(raw)
        assert r.value(b, 1, 'Q') == 0 and r.value(b, 2, 'Q') == 0, 'External buffer unsupported'
    graphs = []
    for sg in r.vector(root, 2, offsets=True):
        tensors = []
        for index, t in enumerate(r.vector(sg, 0, offsets=True)):
            dtype = r.value(t, 1, 'b')
            buf = r.value(t, 2, 'I')
            assert 0 <= buf < len(buffers)
            item = {'index': index, 'name': r.string(t, 3), 'shape': r.vector(t, 0),
                    'shape_signature': r.vector(t, 7), 'type': types.get(dtype, f'UNKNOWN_{dtype}'),
                    'buffer_index': buf, 'stored_buffer_bytes': buffer_lengths[buf],
                    'is_variable': bool(r.value(t, 5, '?'))}
            q = r.pointer(t, 4)
            if q is not None:
                scales, zeros = r.vector(q, 2, 'f'), r.vector(q, 3, 'q')
                item['quantization'] = {'scale_count': len(scales), 'scale_first8': scales[:8],
                    'zero_count': len(zeros), 'zero_first8': zeros[:8],
                    'dimension': r.value(q, 6, 'i'), 'details_type': r.value(q, 4, 'B')}
            tensors.append(item)
        inputs, outputs = r.vector(sg, 1), r.vector(sg, 2)
        assert all(0 <= i < len(tensors) for i in inputs + outputs)
        operators = []
        for index, t in enumerate(r.vector(sg, 3, offsets=True)):
            ci = r.value(t, 0, 'I')
            assert ci < len(codes)
            ins, outs = r.vector(t, 1), r.vector(t, 2)
            assert all(-1 <= i < len(tensors) for i in ins + outs)
            operators.append({'index': index, **codes[ci], 'input_indices': ins, 'output_indices': outs,
                'input_types': [tensors[i]['type'] for i in ins if i >= 0],
                'output_types': [tensors[i]['type'] for i in outs if i >= 0]})
        graphs.append({'name': r.string(sg, 4), 'tensor_count': len(tensors),
            'type_counts': dict(collections.Counter(t['type'] for t in tensors)),
            'inputs': [tensors[i] for i in inputs], 'outputs': [tensors[i] for i in outputs],
            'operator_count': len(operators),
            'operator_counts': dict(collections.Counter(t['name'] for t in operators)),
            'float_type_operator_indices': [t['index'] for t in operators
                if any(x.startswith(('FLOAT', 'BFLOAT')) for x in t['input_types'] + t['output_types'])],
            'variable_tensor_indices': [t['index'] for t in tensors if t['is_variable']],
            'operators': operators, 'tensors': tensors})
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'file_identifier': 'TFL3', 'schema_version': 3, 'description': r.string(root, 3),
        'buffer_count': len(buffers), 'stored_buffer_bytes': sum(buffer_lengths),
        'operator_codes': codes, 'subgraphs': graphs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schema', required=True)
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    assert not output.exists(), 'Preserve previous audit'
    schema = Path(a.schema).read_text()
    rows = [inspect(Path(f), schema) for f in a.files]
    output.write_text(json.dumps({'scope': 'FlatBuffer metadata only; no TensorFlow/LiteRT/import/model loading/invocation, no tensors allocated or numerical benchmark',
        'schema_source': a.schema, 'schema_sha256': hashlib.sha256(schema.encode()).hexdigest(),
        'limits': 'Declared graph types/buffer sizes are not backend integer-only delegation, fidelity, speed, temporal stability or paper-version proof.',
        'files': rows}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{'path': r['path'], 'bytes': r['bytes'], 'graphs': [
        {'inputs': [{'shape': t['shape'], 'type': t['type']} for t in g['inputs']],
         'outputs': [{'shape': t['shape'], 'type': t['type']} for t in g['outputs']],
         'tensors': g['tensor_count'], 'operators': g['operator_count'], 'types': g['type_counts'],
         'float_type_ops': len(g['float_type_operator_indices'])} for g in r['subgraphs']]} for r in rows]))


if __name__ == '__main__':
    main()
