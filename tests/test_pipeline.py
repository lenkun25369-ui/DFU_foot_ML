"""Meaningful local checks. Run: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from dfu import (audit, choose_split, metrics, select_threshold, Letterbox,
                 create_model, image_transform, load_checkpoint, check_files)


class PipelineTests(unittest.TestCase):
    def test_patient_separation_with_repeat_visits_and_mixed_labels(self):
        # Includes patients whose labels change across visits.
        rows = [{'patient_id': f'P{i:03}', 'image_path': f'{i}_{j}.png',
                 'label': (i + j) % 2} for i in range(40) for j in range(2)]
        splits = choose_split(pd.DataFrame(rows))
        self.assertEqual(sum(len(v) for v in splits.values()), 80)
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            self.assertFalse(set(splits[left].patient_id) & set(splits[right].patient_id))
        for part in splits.values():
            self.assertEqual(set(part.label), {0, 1})

    def test_duplicate_pixels_detected_despite_different_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new('RGB', (20, 30), 'red')
            image.save(root / 'a.png', compress_level=0)
            image.save(root / 'b.png', compress_level=9)
            pd.DataFrame([{'image_path': 'a.png', 'patient_id': '001', 'label': 1},
                          {'image_path': 'b.png', 'patient_id': '002', 'label': 0}]).to_csv(root / 'labels.csv', index=False)
            with self.assertRaisesRegex(ValueError, '完全相同'):
                audit(root / 'labels.csv', root)

    def test_patient_id_leading_zeros_preserved_and_file_change_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, colour in [('a', 'red'), ('b', 'blue')]:
                Image.new('RGB', (20, 30), colour).save(root / f'{name}.png')
            pd.DataFrame([{'image_path': 'a.png', 'patient_id': '001', 'label': 1},
                          {'image_path': 'b.png', 'patient_id': '002', 'label': 0}]).to_csv(root / 'labels.csv', index=False)
            frame = audit(root / 'labels.csv', root)
            self.assertEqual(frame.patient_id.tolist(), ['001', '002'])
            check_files(frame, root)
            Image.new('RGB', (20, 30), 'green').save(root / 'a.png')
            with self.assertRaisesRegex(ValueError, '變更'):
                check_files(frame, root)

    def test_validation_threshold_and_classification_direction(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([.1, .8, .6, .9])
        t = select_threshold(y, scores, 'sensitivity', 1.0)
        self.assertAlmostEqual(t, .6)
        result = metrics(y, scores, t)
        self.assertEqual(result['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 0, 'tp': 2})
        self.assertEqual(result['sensitivity'], 1.)
        self.assertEqual(result['specificity'], .5)

    def test_letterbox_preserves_full_wound_context(self):
        im = Image.new('RGB', (200, 100), 'red')
        out = np.asarray(Letterbox(224)(im))
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertTrue(np.all(out[56:168, :, 0] == 255))
        self.assertTrue(np.all(out[0, 0] == np.array([124, 116, 104])))

    def test_saved_model_prediction_and_frozen_bn(self):
        torch.set_num_threads(2)
        model = create_model(False).eval()
        x = image_transform(64)(Image.new('RGB', (60, 40), 'red')).unsqueeze(0)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save({'schema_version': 1, 'architecture': 'efficientnet_b0', 'state_dict': model.state_dict()}, path)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                actual = restored(x)
            torch.testing.assert_close(expected, actual)
        for p in model.features.parameters():
            p.requires_grad = False
        model.train()
        model.features.eval()
        before = {n: v.clone() for n, v in model.features.named_buffers()}
        optimizer = torch.optim.Adam(model.classifier.parameters())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), torch.ones(1))
        loss.backward()
        optimizer.step()
        for name, value in model.features.named_buffers():
            torch.testing.assert_close(before[name], value)


if __name__ == '__main__':
    unittest.main()
