# SPDX-License-Identifier: AGPL-3.0-only
"""Embed external tensors from an authorized Qualcomm v0.64.0 ONNX graph.

Use an existing ONNX export environment. Does not download or publish artifacts.
"""
import argparse
from hashlib import file_digest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.destination.suffix != '.onnx':
        parser.error('Destination must be a new .onnx path')
    with args.graph.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != 'ebd4d22b162cfb3f8113fa48908859a4413c45a82a9ae0bd8a1e5a56f1fb9c5f':
            parser.error('Unexpected Qualcomm landmark graph')
    import onnx
    graph = onnx.load(args.graph, load_external_data=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, args.destination, save_as_external_data=False)
    with args.destination.open('rb') as stream:
        actual = file_digest(stream, 'sha256').hexdigest()
    print('SHA256', actual)
    if actual != 'ac4525a414998f3cf35e3eeed6b76c5f842b7ff4e308f8ece4a4802ac5bac30a':
        raise SystemExit('Byte identity differs; inspect ONNX version/serialization before activation')


if __name__ == '__main__':
    main()
