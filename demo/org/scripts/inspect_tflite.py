#!/usr/bin/env python3
"""用标准库解析 TFLite flatbuffer 的输入/输出、形状与算子。

不做推理，只读模型元数据，便于在无 TensorFlow 环境下核对模型是否
能被 AI Benchmark 的 Custom Model 入口加载。

用法：
    python inspect_tflite.py MobileIE/pretrain/lolv1.tflite
"""

import argparse
import struct
import sys

TYPE_NAMES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8",
    4: "INT64", 5: "STRING", 6: "BOOL", 7: "INT16",
    8: "COMPLEX64", 9: "INT8", 10: "FLOAT64",
}

# 只列出本模型可能用到的 BuiltinOperator（完整枚举见 TFLite schema）
OP_NAMES = {
    0: "ADD", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D",
    9: "FULLY_CONNECTED", 14: "LOGISTIC", 17: "MAX_POOL_2D", 18: "MUL",
    19: "RELU", 22: "RESHAPE", 25: "SOFTMAX", 28: "TANH", 34: "PAD",
    40: "MEAN", 41: "SUB", 42: "DIV", 54: "PRELU", 55: "MAXIMUM",
    59: "NEG", 60: "PADV2", 80: "FAKE_QUANT", 82: "REDUCE_MAX",
    114: "QUANTIZE", 117: "HARD_SWISH",
}


class FlatBuffer:
    def __init__(self, data: bytes):
        self.b = data

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.b, off)[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.b, off)[0]

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.b, off)[0]

    def table(self, pos: int) -> int:
        return pos - self.i32(pos)

    def field(self, table: int, index: int):
        vt = self.table(table)
        vtable_size = self.u16(vt)
        slot = 4 + 2 * index
        if slot >= vtable_size:
            return None
        offset = self.u16(vt + slot)
        return None if offset == 0 else table + offset

    def vector(self, table: int, index: int):
        pos = self.field(table, index)
        if pos is None:
            return None
        start = pos + self.u32(pos)
        length = self.u32(start)
        return start + 4, length

    def table_in_vector(self, start: int, index: int) -> int:
        elem = start + 4 * index
        return elem + self.u32(elem)

    def string(self, table: int, index: int):
        vec = self.vector(table, index)
        if not vec:
            return None
        start, length = vec
        return self.b[start:start + length].decode("utf-8", errors="replace")

    def ints(self, table: int, index: int):
        vec = self.vector(table, index)
        if not vec:
            return None
        start, length = vec
        return list(struct.unpack_from("<%di" % length, self.b, start))


def describe_tensor(fb: FlatBuffer, tensors_start: int, index: int):
    t = fb.table_in_vector(tensors_start, index)
    type_pos = fb.field(t, 1)
    return {
        "name": fb.string(t, 3),
        "shape": fb.ints(t, 0),
        # TensorType 默认值是 FLOAT32(0)，FlatBuffers 会省略该字段
        "dtype": TYPE_NAMES.get(fb.b[type_pos] if type_pos is not None else 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to .tflite")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        data = f.read()

    fb = FlatBuffer(data)
    model = fb.u32(0)
    subgraphs = fb.vector(model, 2)
    if not subgraphs:
        sys.exit("不是有效的 TFLite 模型：未找到 subgraph")

    sg_start, sg_count = subgraphs
    sg = fb.table_in_vector(sg_start, 0)
    tensors = fb.vector(sg, 0)
    inputs = fb.vector(sg, 1)
    outputs = fb.vector(sg, 2)

    print(f"file      : {args.model}")
    print(f"size      : {len(data)} bytes")
    print(f"subgraphs : {sg_count}")
    print(f"tensors   : {tensors[1] if tensors else 0}")

    for label, vec in (("INPUT", inputs), ("OUTPUT", outputs)):
        if not vec:
            continue
        start, count = vec
        order = struct.unpack_from("<%di" % count, data, start)
        print(f"\n{label} ({count}):")
        for idx in order:
            info = describe_tensor(fb, tensors[0], idx)
            print(f"  [{idx}] {info['name']} shape={info['shape']} dtype={info['dtype']}")

    opcodes = fb.vector(model, 1)
    ops = fb.vector(sg, 3)
    print(f"\noperators : {ops[1] if ops else 0}")
    if opcodes and ops:
        opcode_start, opcode_count = opcodes
        counts = {}
        for i in range(ops[1]):
            op = fb.table_in_vector(ops[0], i)
            code_pos = fb.field(op, 0)
            # opcode_index 默认值 0，可能被省略
            code = fb.u32(code_pos) if code_pos is not None else 0
            if code >= opcode_count:
                counts[code] = counts.get(code, 0) + 1
                continue
            entry = fb.table_in_vector(opcode_start, code)
            builtin_pos = fb.field(entry, 3)
            # builtin_code 默认值 0 (ADD)，同样可能被省略
            builtin = fb.i32(builtin_pos) if builtin_pos is not None else 0
            counts[builtin] = counts.get(builtin, 0) + 1
        for builtin, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {OP_NAMES.get(builtin, 'builtin_%d' % builtin):<20} x{n}")


if __name__ == "__main__":
    main()
