"""Exercise real checkpoint serialization/reading using tiny synthetic records."""

import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import zstandard

from xfold.params import (RecordError, _MultiFileIO, encode_record, get_alphafold3_params,
                          open_for_reading, read_records, select_model_files)


class ParamsRuntimeTests(unittest.TestCase):
    def test_record_roundtrip_dtypes_shapes_and_multiple_records(self):
        arrays = (np.arange(15, dtype=np.float32).reshape(3, 5).T,
                  np.array(3, dtype=np.float16), np.arange(16, dtype=np.uint8))
        payload = b''.join(encode_record('scope', str(i), array) for i, array in enumerate(arrays))
        received = list(read_records(io.BytesIO(payload)))
        self.assertEqual(len(received), len(arrays))
        for i, (scope, name, tensor) in enumerate(received):
            self.assertEqual((scope, name), ('scope', str(i)))
            torch.testing.assert_close(tensor, torch.from_numpy(arrays[i].copy()), rtol=0, atol=0)

    def test_truncated_records_fail_clearly(self):
        record = encode_record('scope', 'weight', np.ones((3, 5), dtype=np.float32))
        for truncated in (record[:7], record[:-1]):
            with self.assertRaises(RecordError):
                list(read_records(io.BytesIO(truncated)))

    def test_compressed_and_uncompressed_checkpoint_loading(self):
        value = np.arange(21, dtype=np.float32).reshape(3, 7)
        payload = encode_record('scope', 'weight', value)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.bin', '.bin.zst'):
                path = Path(directory, 'af3' + suffix)
                path.write_bytes(zstandard.ZstdCompressor().compress(payload) if suffix.endswith('zst') else payload)
                loaded = get_alphafold3_params(path)
                torch.testing.assert_close(loaded['scope/weight'], torch.from_numpy(value), rtol=0, atol=0)

    def test_multifile_read_seek_and_eof_follow_file_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, f'part{i}') for i in range(3)]
            for path, payload in zip(paths, (b'ab', b'cde', b'fgh')):
                path.write_bytes(payload)
            with _MultiFileIO(paths) as stream:
                self.assertEqual(stream.read(), b'abcdefgh')
                self.assertEqual(stream.read(), b'')
                self.assertEqual(stream.seek(-3, io.SEEK_END), 5)
                self.assertEqual(stream.read(), b'fgh')
                self.assertEqual(stream.seek(1), 1)
                self.assertEqual(stream.read(4), b'bcde')
                self.assertEqual(stream.seek(-2, io.SEEK_CUR), 3)
                self.assertEqual(stream.read(), b'defgh')
                self.assertEqual(stream.seek(100), 100)
                self.assertEqual(stream.read(), b'')
                with self.assertRaises(ValueError):
                    stream.seek(-1)
            self.assertTrue(stream.closed)
            self.assertTrue(all(handle.closed for handle in stream._handles))
            with self.assertRaises(ValueError):
                stream.read(1)

    def test_checkpoint_part_selection_accepts_both_naming_conventions(self):
        for suffix in ('.0.bin', '.bin.0', '.0.bin.zst', '.bin.zst.0'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory, 'af3' + suffix)
                path.touch()
                files, compressed = select_model_files(Path(directory), 'af3')
                self.assertEqual(files, [path])
                self.assertEqual(compressed, 'zst' in suffix)

    def test_checkpoint_parts_are_in_numeric_order_and_read_across_boundaries(self):
        record = encode_record('scope', 'w', np.arange(30, dtype=np.float32))
        for compressed in (False, True):
            payload = zstandard.ZstdCompressor().compress(record) if compressed else record
            with tempfile.TemporaryDirectory() as directory:
                pieces = [payload[i*len(payload)//12:(i+1)*len(payload)//12] for i in range(12)]
                suffix = '.bin.zst' if compressed else '.bin'
                paths = [Path(directory, f'af3{suffix}.{i}') for i in range(12)]
                for path, piece in zip(paths, pieces):
                    path.write_bytes(piece)
                selected, actual_compressed = select_model_files(Path(directory))
                self.assertEqual(selected, paths)
                self.assertEqual(actual_compressed, compressed)
                with open_for_reading(selected, compressed) as stream:
                    received = list(read_records(stream))
                self.assertEqual(len(received), 1)
                torch.testing.assert_close(received[0][2], torch.arange(30, dtype=torch.float32))

    def test_empty_parts_and_no_files_do_not_leak_or_loop(self):
        with self.assertRaisesRegex(ValueError, 'At least one'):
            _MultiFileIO([])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, str(i)) for i in range(4)]
            for path, value in zip(paths, (b'', b'ab', b'', b'')):
                path.write_bytes(value)
            with _MultiFileIO(paths) as stream:
                self.assertEqual(stream.read(), b'ab')
                self.assertEqual(stream.read(), b'')


if __name__ == '__main__':
    unittest.main()
