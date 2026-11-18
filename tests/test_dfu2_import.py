"""Critical data-adaptation regression tests; fixtures contain no patient data."""
import argparse
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import Workbook
from PIL import Image

from import_dfu2 import read_labels, resolve_duplicates, apply_patient_map, run, canonical_key
from dfu import load_splits, summary


class DFU2ImportTests(unittest.TestCase):
    def make_workbook(self, root):
        path = root / 'labels.xlsx'
        book = Workbook()
        sheet = book.active
        sheet.append(['ï»¿split', 'image_name', 'zip_path', 'label'])
        for split, folder, offset in [('train', 'train', 0), ('val', 'validation', 2), ('test', 'test', 4)]:
            for label in (0, 1):
                name = f'{offset + label:04}.png'
                sheet.append([split, name, f'garbled_x0096_/DFU2/{folder}/images/{name}', float(label)])
        sheet.append(['train', '9999.png', 'garbled/DFU2/train/images/9999.png', '?'])
        book.save(path)
        return path

    def test_excel_bom_split_and_unknown_and_invalid_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_workbook(root)
            frame, _ = read_labels(path)
            self.assertEqual(len(frame), 7)
            self.assertEqual(frame.image_name.iloc[0], '0000.png')
            self.assertEqual(int(frame.label.isna().sum()), 1)
            self.assertTrue(frame.patient_id.eq('').all())
            from openpyxl import load_workbook
            book = load_workbook(path)
            book.active['D2'] = 2
            book.save(path)
            with self.assertRaisesRegex(ValueError, '只接受'):
                read_labels(path)

    def test_dedup_keeps_holdout_and_rejects_conflicting_labels(self):
        frame = pd.DataFrame([
            {'source_row': 2, 'split': 'train', 'image_name': 'a.png', 'image_path': 'a', 'pixel_sha256': 'x', 'label': 1},
            {'source_row': 3, 'split': 'val', 'image_name': 'b.png', 'image_path': 'b', 'pixel_sha256': 'x', 'label': 1},
            {'source_row': 4, 'split': 'test', 'image_name': 'c.png', 'image_path': 'c', 'pixel_sha256': 'x', 'label': 1},
            {'source_row': 5, 'split': 'test', 'image_name': 'd.png', 'image_path': 'd', 'pixel_sha256': 'y', 'label': None},
        ])
        clean, excluded = resolve_duplicates(frame)
        self.assertEqual(clean.image_name.tolist(), ['c.png'])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(excluded.kept_split.eq('test').all())
        frame.loc[0, 'label'] = 0
        with self.assertRaisesRegex(ValueError, '衝突'):
            resolve_duplicates(frame)

    def test_patient_map_requires_complete_real_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, _ = read_labels(self.make_workbook(root))
            path = root / 'patients.csv'
            mapping = frame[['split', 'image_name', 'patient_id']].copy()
            mapping.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, '所有已標記'):
                apply_patient_map(frame, path)
            mapping.patient_id = ['001', '002', '003', '004', '005', '006', '007']
            mapping.to_csv(path, index=False)
            result = apply_patient_map(frame, path)
            self.assertEqual(result.patient_id.iloc[0], '001')
            mapping.loc[2, 'patient_id'] = '001'
            mapping.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, '同一病人跨組'):
                apply_patient_map(frame, path)

    def test_zip_import_ignores_masks_and_roundtrip_without_fake_patient_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excel = self.make_workbook(root)
            frame, _ = read_labels(excel)
            archive = root / 'source.zip'
            folders = {'train': 'train', 'val': 'validation', 'test': 'test'}
            with ZipFile(archive, 'w') as z:
                for i, row in frame.iterrows():
                    source = root / 'fixture.png'
                    Image.new('RGB', (20, 30), (i * 25, 10, 20)).save(source)
                    z.write(source, f'wrapper/DFU2/{folders[row["split"]]}/images/{row.image_name}')
                z.writestr('DFU2/train/labels/0000.png', b'not an image')
                z.writestr('DFU3/images/0000.png', b'not an image')
            args = argparse.Namespace(excel=excel, sheet=None, source_zip=archive, source_dir=None,
                                      data_root=root / 'data', out=root / 'splits', patient_map=None,
                                      label_definition='test labels')
            run(args)
            parts, manifest = load_splits(args.out)
            self.assertFalse(manifest['patient_separation_verified'])
            self.assertEqual([len(parts[x]) for x in ('train', 'val', 'test')], [2, 2, 2])
            self.assertTrue(all(p.patient_id.eq('').all() for p in parts.values()))
            self.assertIsNone(summary(parts['train'])['patients'])
            self.assertFalse((args.data_root / 'DFU2/train/labels').exists())
            self.assertEqual(len(pd.read_csv(args.out / 'excluded_unknown.csv')), 1)
            # Source remains unchanged; rerun cannot silently overwrite research split.
            with self.assertRaises(FileExistsError):
                run(args)
            # Incorrect claims of patient verification cannot survive a reload.
            import json
            manifest['patient_separation_verified'] = True
            (args.out / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'patient_id 有空白'):
                load_splits(args.out)

    def test_paths_cannot_traverse_outside_root(self):
        with self.assertRaisesRegex(ValueError, '不能包含'):
            canonical_key('../DFU2/train/images/a.png')
        self.assertIsNone(canonical_key('DFU2/train/labels/a.png'))


if __name__ == '__main__':
    unittest.main()
