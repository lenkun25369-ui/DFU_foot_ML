"""EfficientNet-B0 referral classification with auxiliary wound segmentation.

Place this file beside the existing dfu.py. Use the same dfu_referral environment.
No new dependencies. Original splits and baseline checkpoint remain unchanged.

1. Prepare masks for the INCLUDED train/val images only (700/174 for this dataset):
 python dfu_multitask.py prepare-masks --source-zip raw_data/drive-download-20261111T024845Z-1-001.zip --data-root data --splits splits_dfu2 --out masks_dfu2
2. Train from ImageNet initialization, not from the previous referral model:
 python dfu_multitask.py train --data-root data --splits splits_dfu2 --masks masks_dfu2 --out runs/b0_mtl_v1 --device cuda --batch-size 8 --seg-weight 0.5
3. Compare on validation ONLY, before any further test evaluation:
 python dfu_multitask.py compare --baseline runs/b0_dfu2_v1 --candidate runs/b0_mtl_v1 --splits splits_dfu2 --specificity 0.80 --out comparisons/b0_mtl_v1
4. Predict a new image (no ground-truth mask required):
 python dfu_multitask.py predict --checkpoint runs/b0_mtl_v1/model.pt --image data/new_foot.jpg
5. Optional descriptive evaluation on the already-inspected test split:
 python dfu_multitask.py evaluate --checkpoint runs/b0_mtl_v1/model.pt --data-root data --splits splits_dfu2 --device cuda --batch-size 8

Method: shared ImageNet EfficientNet-B0 encoder; original global-pooling
classification head; small U-Net-style auxiliary decoder with skip connections
and GroupNorm. Loss = classification BCE + seg_weight * (pixel BCE + soft Dice
loss). No ground-truth mask is fed into the image encoder or classification head.
Masks and images share resize/padding/flip geometry; masks use nearest-neighbor.
0/1 masks are normalized; grayscale masks are binarized at 128. Empty masks
are treated as unverified segmentation annotations: their images retain the
classification target but do not contribute segmentation loss or Dice. This
dataset has 11 such training masks and 3 validation masks. Gray pixels and
empty masks are listed in the preparation audit; original files are preserved.
Segmentation losses ignore artificial padding. All original image context remains.

Head stage (5 epochs): encoder frozen/eval; classification head + decoder train.
Finetune (up to 30 epochs): all parameters train, LR 1e-5. Select checkpoint and
stop on validation CLASSIFICATION AP, not Dice. Threshold is validation-only.
Default seg_weight=0.5 and comparison specificity=0.80 are experimental choices,
not clinically validated targets. For an architecture-matched control, repeat
train with --seg-weight 0 --out runs/b0_mtl_control (same other settings).

New checkpoints use a different architecture; use THIS script to load them.
The original dfu.py/app.py do not support these multitask checkpoints. Results
CSVs can still be inspected using review_errors.py. Training does not resume an
interrupted run: use a new output directory. No patient IDs are invented. Prior
test results have already been reviewed, so repeat test comparisons are
exploratory and require fresh independent validation before stronger claims.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

try:
    import dfu as base
except ImportError as exc:
    raise SystemExit('請把 dfu_multitask.py 放到原本 dfu.py 的同一個資料夾。') from exc

ARCHITECTURE = 'efficientnet_b0_auxseg_v1'
TAPS = (1, 2, 3, 5, 8)
MASK_FIELDS = ['image_path', 'file_sha256', 'mask_path', 'mask_sha256', 'mask_pixel_sha256', 'width', 'height', 'seg_supervision', 'mask_gray_pixels']


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / str(relative).replace('\\', '/')).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'路徑超出 data-root：{relative}')
    return path


def read_mask(source, return_info=False):
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        a = np.asarray(im)
    if not np.array_equal(a[..., 0], a[..., 1]) or not np.array_equal(a[..., 1], a[..., 2]):
        raise ValueError('遮罩不是灰階二值影像，請核對類別定義。')
    a = a[..., 0]
    values = set(np.unique(a).tolist())
    binary01 = values <= {0, 1}
    binary = a > 0 if binary01 else a >= 128
    info = {'source_values': sorted(values), 'gray_pixels': 0 if binary01 else int(((a != 0) & (a != 255)).sum()),
            'empty': not bool(binary.any())}
    mask = Image.fromarray(binary.astype(np.uint8) * 255)
    return (mask, info) if return_info else mask


def mask_table_hash(frame):
    records = frame[MASK_FIELDS].astype(str).sort_values('image_path').to_dict('records')
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def mask_key(path):
    parts = PurePosixPath(str(path).replace('\\', '/')).parts
    if '..' in parts:
        raise ValueError('遮罩路徑含有 ..')
    for i, part in enumerate(parts):
        tail = parts[i:]
        if part == 'DFU2' and len(tail) == 4 and tail[1] in ('train', 'validation') and tail[2] == 'labels':
            return '/'.join(tail)
    return None


def prepare_masks(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在，請使用另一個 --out。')
    parts, _ = base.load_splits(args.splits)
    for split in ('train', 'val'):
        base.check_files(parts[split], args.data_root)
    prepared, audit_rows = {}, []
    with ZipFile(args.source_zip) as archive:
        index = {}
        for name in archive.namelist():
            key = mask_key(name)
            if key:
                if key in index:
                    raise ValueError(f'ZIP 有多個相同遮罩路徑：{key}')
                index[key] = name
        for split in ('train', 'val'):
            frame = parts[split].copy()
            frame['seg_supervision'] = 1
            frame['mask_gray_pixels'] = 0
            for count, (i, row) in enumerate(frame.iterrows(), 1):
                image_parts = PurePosixPath(row.image_path).parts
                folder = 'train' if split == 'train' else 'validation'
                if len(image_parts) != 4 or image_parts[:3] != ('DFU2', folder, 'images'):
                    raise ValueError('這個匯入器需要原 DFU2 split/images 檔案路徑。')
                relative = f'DFU2/{folder}/labels/{image_parts[-1]}'
                if relative not in index:
                    raise FileNotFoundError(f'缺少遮罩：{relative}')
                data = archive.read(index[relative])
                mask, info = read_mask(io.BytesIO(data), return_info=True)
                image = base.read_rgb(safe_path(args.data_root, row.image_path))
                if mask.size != image.size:
                    raise ValueError(f'影像／遮罩尺寸不符：{row.image_path}')
                destination = safe_path(args.data_root, relative)
                if destination.exists() and destination.read_bytes() != data:
                    raise FileExistsError(f'已有不同內容的遮罩，不覆寫：{destination}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(data)
                frame.loc[i, 'mask_path'] = relative
                frame.loc[i, 'mask_sha256'] = hashlib.sha256(data).hexdigest()
                frame.loc[i, 'mask_pixel_sha256'] = hashlib.sha256(str(mask.size).encode() + mask.tobytes()).hexdigest()
                frame.loc[i, 'mask_foreground_fraction'] = float((np.asarray(mask) > 0).mean())
                frame.loc[i, 'seg_supervision'] = int(not info['empty'])
                frame.loc[i, 'mask_gray_pixels'] = info['gray_pixels']
                if info['empty'] or info['gray_pixels']:
                    audit_rows.append({'split': split, 'image_path': row.image_path, 'mask_path': relative,
                                       'label': int(row.label), 'source_values': str(info['source_values']),
                                       'gray_pixels': info['gray_pixels'], 'seg_supervision': int(not info['empty']),
                                       'action': 'skip_segmentation_keep_classification' if info['empty'] else 'binarize_at_128'})
                if count % 100 == 0 or count == len(frame):
                    print(f'{split}: {count}/{len(frame)} masks checked', flush=True)
            prepared[split] = frame
    out.mkdir(parents=True)
    manifest = {'schema_version': 1, 'source_zip_sha256': base.sha_file(args.source_zip),
                'split_manifest_sha256': base.sha_file(Path(args.splits) / 'manifest.json'),
                'mask_definition': '0/1 normalized; grayscale>=128 wound; lower values background',
                'mask_processing_version': 'binary128_empty_skip_v1',
                'splits': {}}
    for split, frame in prepared.items():
        frame.to_csv(out / f'{split}.csv', index=False, encoding='utf-8-sig')
        manifest['splits'][split] = {'images': len(frame), 'table_sha256': mask_table_hash(frame),
                                    'empty_masks_skipped_for_segmentation': int(frame.seg_supervision.eq(0).sum()),
                                    'seg_supervised_images': int(frame.seg_supervision.sum()),
                                    'masks_with_gray_pixels': int(frame.mask_gray_pixels.gt(0).sum())}
    pd.DataFrame(audit_rows, columns=['split', 'image_path', 'mask_path', 'label', 'source_values',
                                     'gray_pixels', 'seg_supervision', 'action']).to_csv(out/'mask_audit.csv', index=False, encoding='utf-8-sig')
    base.json_save(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('完成：只準備原切分保留的 train/val 遮罩；test 不需遮罩。')


def load_mask_splits(directory, split_directory, data_root):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('mask_processing_version') != 'binary128_empty_skip_v1':
        raise ValueError('遮罩處理版本不符，請重新 prepare-masks。')
    parts, original = base.load_splits(split_directory, manifest['split_manifest_sha256'])
    result = {}
    for split in ('train', 'val'):
        frame = pd.read_csv(directory / f'{split}.csv', dtype={'patient_id': str}, keep_default_na=False, encoding='utf-8-sig')
        if frame.image_path.duplicated().any() or mask_table_hash(frame) != manifest['splits'][split]['table_sha256']:
            raise ValueError(f'{split} 遮罩清單與 manifest 不符。')
        if base.table_hash(frame) != base.table_hash(parts[split]):
            raise ValueError(f'{split} 的影像、標籤或病人資訊與原切分不符。')
        if not frame.seg_supervision.isin([0, 1]).all() or not frame.seg_supervision.any():
            raise ValueError(f'{split} 需要至少一張有效分割標註。')
        # Match original row order for labels/scores throughout validation.
        frame = frame.set_index('image_path').loc[parts[split].image_path].reset_index()
        base.check_files(frame, data_root)
        for row in frame.itertuples():
            path = safe_path(data_root, row.mask_path)
            if not path.is_file() or base.sha_file(path) != row.mask_sha256:
                raise ValueError(f'遮罩遺失或已改變：{path}')
        result[split] = frame
    return result, original


def paired_transform(image, mask, size, training=False, flip=None):
    if image.size != mask.size:
        raise ValueError('影像與遮罩尺寸不同。')
    valid = Image.new('L', image.size, 255)
    image = base.Letterbox(size)(image)
    mask = ImageOps.pad(mask, (size, size), method=Image.Resampling.NEAREST, color=0, centering=(.5, .5))
    valid = ImageOps.pad(valid, (size, size), method=Image.Resampling.NEAREST, color=0, centering=(.5, .5))
    flip = bool(torch.rand(()) < .5) if training and flip is None else bool(flip)
    if flip:
        image, mask, valid = (ImageOps.mirror(v) for v in (image, mask, valid))
    image = T.functional.normalize(T.functional.to_tensor(image), [.485, .456, .406], [.229, .224, .225])
    return image, T.functional.to_tensor(mask), T.functional.to_tensor(valid)


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, frame, root, size=224, training=False):
        self.frame, self.root, self.size, self.training = frame.reset_index(drop=True), root, size, training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = base.read_rgb(safe_path(self.root, row.image_path))
        mask = read_mask(safe_path(self.root, row.mask_path))
        image, mask, valid = paired_transform(image, mask, self.size, self.training)
        if not int(row.seg_supervision):
            valid.zero_()
        return image, torch.tensor(float(row.label), dtype=torch.float32), mask, valid


class DecoderBlock(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming + skip, outgoing, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, outgoing), nn.SiLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, outgoing), nn.SiLU())

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.layers(torch.cat([x, skip], dim=1))


class MultiTaskB0(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        model = base.create_model(pretrained)
        self.features, self.avgpool, self.classifier = model.features, model.avgpool, model.classifier
        # Decoder is constructed without advancing the classification RNG stream;
        # lambda=0 can use the original seed/shuffle/dropout sequence as a control.
        with torch.random.fork_rng(devices=[]):
            self.bottleneck = nn.Sequential(nn.Conv2d(1280, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.SiLU())
            self.decoder = nn.ModuleList([DecoderBlock(128, 112, 128), DecoderBlock(128, 40, 64),
                                          DecoderBlock(64, 24, 48), DecoderBlock(48, 16, 32)])
            self.segmentation_head = nn.Conv2d(32, 1, 1)

    def forward(self, image, return_seg=False):
        x, taps = image, []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_seg and i in TAPS:
                taps.append(x)
        classification = self.classifier(self.avgpool(x).flatten(1)).flatten()
        if not return_seg:
            return classification
        mask = self.bottleneck(taps[-1])
        for layer, skip in zip(self.decoder, reversed(taps[:-1])):
            mask = layer(mask, skip)
        mask = F.interpolate(self.segmentation_head(mask), size=image.shape[-2:], mode='bilinear', align_corners=False)
        return classification, mask


def segmentation_loss(logits, target, valid):
    dims = (1, 2, 3)
    count = valid.sum(dims).clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * valid).sum(dims) / count
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    dice = (2 * (probability * target).sum(dims) + 1) / (probability.sum(dims) + target.sum(dims) + 1)
    active = valid.sum(dims) > 0
    return (bce + 1 - dice)[active].mean() if active.any() else logits.sum() * 0


def hard_dice(logits, target, valid):
    dims = (1, 2, 3)
    prediction = (logits >= 0).float() * valid
    target = target * valid
    denominator = prediction.sum(dims) + target.sum(dims)
    return torch.where(denominator > 0, 2 * (prediction * target).sum(dims) / denominator.clamp_min(1), torch.ones_like(denominator))


def make_loader(frame, root, size, batch, workers=0, training=False, seed=42):
    return torch.utils.data.DataLoader(MaskDataset(frame, root, size, training), batch_size=batch,
        shuffle=training, num_workers=workers, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed))


def validate(model, loader, device):
    model.eval()
    scores, total_cls, total_seg, total_dice, n, n_seg = [], 0., 0., 0., 0, 0
    with torch.inference_mode():
        for x, y, mask, valid in loader:
            x, y, mask, valid = (v.to(device) for v in (x, y, mask, valid))
            logits, segmentation = model(x, return_seg=True)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            total_cls += float(F.binary_cross_entropy_with_logits(logits, y)) * len(y)
            active = valid.sum((1, 2, 3)) > 0
            count = int(active.sum())
            total_seg += float(segmentation_loss(segmentation, mask, valid)) * count
            total_dice += float(hard_dice(segmentation, mask, valid)[active].sum())
            n += len(y); n_seg += count
    return np.asarray(scores), {'val_loss': total_cls/n, 'val_seg_loss': total_seg/n_seg if n_seg else None,
                               'val_dice': total_dice/n_seg if n_seg else None, 'val_seg_images': n_seg}


def load_checkpoint(path, device='cpu', require_final=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('schema_version') != 1:
        raise ValueError('請使用 dfu_multitask.py 產生的模型檔。')
    if require_final and checkpoint.get('threshold_policy') == 'not_yet_selected':
        raise ValueError('這輪訓練還沒完成分類門檻選擇。')
    model = MultiTaskB0(False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def train(args):
    import torchvision
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在；請指定新的 --out。')
    if args.head_epochs < 1 or args.finetune_epochs < 0 or args.patience < 1 or not np.isfinite(args.seg_weight) or args.seg_weight < 0:
        raise ValueError('epochs/patience 不合法，或 seg-weight 不是非負有限值。')
    if args.image_size < 64 or args.image_size % 32 or args.batch_size < 1:
        raise ValueError('image-size 需 >=64 且為 32 的倍數；batch-size 需 >=1。')
    if args.threshold_policy == 'sensitivity' and not 0 < args.target_sensitivity <= 1:
        raise ValueError('target-sensitivity 必須在 (0,1]。')
    base.seed_everything(args.seed, args.threads)
    parts, manifest = load_mask_splits(args.masks, args.splits, args.data_root)
    if not manifest.get('patient_separation_verified', True):
        print('探索性影像切分：沒有病人 ID，新病人泛化能力尚未確認。', flush=True)
    device = base.device_for(args.device)
    print(f'Device: {device}; train={len(parts["train"])}, val={len(parts["val"])}; no test images/masks loaded.', flush=True)
    model = MultiTaskB0(not args.random_init).to(device)
    out.mkdir(parents=True)
    config = {k: v for k, v in vars(args).items() if k != 'func'}
    config.update({'architecture': ARCHITECTURE, 'loss_definition': 'cls_BCE + seg_weight*(pixel_BCE+soft_Dice_loss)',
                   'split_summary': manifest['splits'], 'patient_separation_verified': manifest.get('patient_separation_verified', True),
                   'versions': {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__)}})
    base.json_save(out/'config.json', config)
    train_loader = make_loader(parts['train'], args.data_root, args.image_size, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(parts['val'], args.data_root, args.image_size, args.batch_size, args.workers)
    pos = int(parts['train'].label.sum())
    pos_weight = (len(parts['train']) - pos)/pos if args.class_weight else 1.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history, best_ap, epoch = [], -1., 0
    checkpoint_path = out/'model.pt'
    for stage, epochs, lr in [('head', args.head_epochs, args.head_lr), ('finetune', args.finetune_epochs, args.finetune_lr)]:
        if not epochs:
            continue
        for parameter in model.features.parameters():
            parameter.requires_grad = stage == 'finetune'
        if stage == 'finetune':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['state_dict'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        bad, stage_best = 0, -1.
        for _ in range(epochs):
            epoch += 1
            model.train()
            if stage == 'head':
                model.features.eval()
            total_cls = total_seg = total_loss = seen = seg_seen = 0
            for x, y, mask, valid in train_loader:
                x, y, mask, valid = (v.to(device) for v in (x, y, mask, valid))
                optimizer.zero_grad(set_to_none=True)
                logits, seg = model(x, return_seg=True)
                cls_loss = criterion(logits, y)
                seg_loss = segmentation_loss(seg, mask, valid)
                loss = cls_loss + args.seg_weight * seg_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss 出現 NaN/Inf；停止訓練。')
                loss.backward()
                optimizer.step()
                total_cls += float(cls_loss.detach()) * len(y)
                n_active = int((valid.sum((1, 2, 3)) > 0).sum())
                total_seg += float(seg_loss.detach()) * n_active
                seg_seen += n_active
                total_loss += float(loss.detach()) * len(y)
                seen += len(y)
            scores, segmentation_metrics = validate(model, val_loader, device)
            classification_metrics = base.metrics(parts['val'].label, scores, .5)
            row = {'epoch': epoch, 'stage': stage, 'train_loss': total_cls/seen,
                   'train_seg_loss': total_seg/seg_seen if seg_seen else None, 'train_total_loss': total_loss/seen,
                   **segmentation_metrics, 'val_auroc': classification_metrics['auroc'],
                   'val_average_precision': classification_metrics['average_precision']}
            history.append(row)
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            print(json.dumps(row), flush=True)
            ap = row['val_average_precision']
            if ap > best_ap + 1e-8:
                best_ap = ap
                checkpoint = {'schema_version': 1, 'architecture': ARCHITECTURE,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'image_size': args.image_size, 'preprocessing': 'exif_rgb_letterbox_imagenet_v1',
                    'threshold': .5, 'threshold_policy': 'not_yet_selected', 'epoch': epoch,
                    'split_manifest_sha256': base.sha_file(Path(args.splits)/'manifest.json'),
                    'mask_manifest_sha256': base.sha_file(Path(args.masks)/'manifest.json'),
                    'label_definition': manifest['label_definition'], 'positive_label': 1,
                    'patient_separation_verified': manifest.get('patient_separation_verified', True),
                    'initialization': 'random_test_only' if args.random_init else 'imagenet1k_v1',
                    'seg_weight': args.seg_weight, 'val_average_precision': best_ap,
                    'seed': args.seed, 'pos_weight': pos_weight, 'versions': config['versions']}
                torch.save(checkpoint, checkpoint_path)
            if ap > stage_best + 1e-8:
                stage_best, bad = ap, 0
            else:
                bad += 1
            if stage == 'finetune' and bad >= args.patience:
                print('Early stopping: validation classification AP stopped improving.', flush=True)
                break
    model, checkpoint = load_checkpoint(checkpoint_path, device, require_final=False)
    scores, segmentation_metrics = validate(model, val_loader, device)
    threshold = base.select_threshold(parts['val'].label.to_numpy(), scores, args.threshold_policy, args.target_sensitivity)
    checkpoint.update({'threshold': threshold, 'threshold_policy': args.threshold_policy,
                       'target_sensitivity': args.target_sensitivity if args.threshold_policy == 'sensitivity' else None})
    torch.save(checkpoint, checkpoint_path)
    base.json_save(out/'validation_metrics.json', {'at_0_5': base.metrics(parts['val'].label, scores, .5),
        'at_selected_threshold': base.metrics(parts['val'].label, scores, threshold),
        'segmentation': {**segmentation_metrics, 'dice_definition': 'Mean per-image hard Dice at 0.5; unverified empty source masks and padding excluded'},
        'note': 'Validation was used for model/threshold selection; not an unbiased final estimate.'})
    predictions = parts['val'].copy()
    predictions['score'], predictions['prediction'] = scores, (scores >= threshold).astype(int)
    predictions.to_csv(out/'validation_predictions.csv', index=False, encoding='utf-8-sig')
    plot_history(pd.DataFrame(history), out/'training_curves.png')
    print(f'完成：{checkpoint_path}；最佳 epoch={checkpoint["epoch"]}；驗證門檻={threshold:.6f}。')
    print('下一步使用 compare 比較驗證結果；不需重新訓練第一版。')


def plot_history(frame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name in ['train_loss', 'val_loss']:
        axes[0, 0].plot(frame.epoch, frame[name], label=name)
    axes[0, 0].set_title('Classification BCE (train weighted if enabled)')
    for name in ['val_auroc', 'val_average_precision']:
        axes[0, 1].plot(frame.epoch, frame[name], label=name)
    axes[0, 1].set(title='Validation referral scores', ylim=(0, 1))
    for name in ['train_seg_loss', 'val_seg_loss']:
        axes[1, 0].plot(frame.epoch, frame[name], label=name)
    axes[1, 0].set_title('Segmentation BCE + soft Dice loss')
    axes[1, 1].plot(frame.epoch, frame.val_dice, label='val_dice')
    axes[1, 1].set(title='Validation wound Dice (not referral accuracy)', ylim=(0, 1))
    for ax in axes.flat:
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def evaluate(args):
    base.seed_everything(args.seed, args.threads)
    device = base.device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    parts, manifest = base.load_splits(args.splits, checkpoint['split_manifest_sha256'])
    frame = parts['test']
    base.check_files(frame, args.data_root)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent/'test'
    if out.exists():
        raise FileExistsError(f'{out} 已存在。請保留結果；如要重現，明確指定另一個 --out。')
    scores = base.get_scores(model, base.loader_for(frame, args.data_root, checkpoint['image_size'], args.batch_size), device)
    threshold = checkpoint['threshold']
    verified = manifest.get('patient_separation_verified', True)
    report = {'unit': 'image', 'patients': int(frame.patient_id.nunique()) if verified else None,
        'patient_separation_verified': verified, 'checkpoint_sha256': base.sha_file(args.checkpoint),
        'label_definition': checkpoint['label_definition'], 'ground_truth_masks_used': False,
        'at_selected_threshold': base.metrics(frame.label, scores, threshold),
        'at_0_5': base.metrics(frame.label, scores, .5),
        'ci95_patient_cluster_bootstrap': base.cluster_bootstrap(frame, scores, threshold, args.bootstrap, args.seed) if verified else None,
        'ci_note': 'Patient-cluster intervals only when patient IDs are available; otherwise independence is unverified.',
        'evaluation_note': 'This DFU2 test set was already reviewed during baseline development; subsequent comparisons are exploratory.',
        'average_precision_note': 'AP is average precision, not trapezoidal PR-AUC.'}
    out.mkdir(parents=True)
    base.json_save(out/'metrics.json', report)
    predictions = frame.copy()
    predictions['score'], predictions['prediction'] = scores, (scores >= threshold).astype(int)
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    predictions.loc[(predictions.label == 1) & (predictions.prediction == 0)].to_csv(out/'false_negatives.csv', index=False, encoding='utf-8-sig')
    base.plot_evaluation(frame.label, scores, threshold, out/'evaluation.png')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def predict(args):
    torch.set_num_threads(args.threads)
    device = base.device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    result = base.predict_image(model, checkpoint, args.image, device)
    result['ground_truth_mask_required'] = False
    if args.out:
        if Path(args.out).exists():
            raise FileExistsError(f'{args.out} 已存在。')
        base.json_save(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def operating_point(y, scores, specificity):
    # Validation-only comparison. Thresholds are not written back to checkpoints.
    candidates = np.unique(np.r_[0., scores, np.nextafter(max(scores), np.inf)])
    candidates = [base.metrics(y, scores, t) for t in candidates]
    possible = [m for m in candidates if m['specificity'] >= specificity - 1e-12]
    return max(possible, key=lambda m: (m['sensitivity'], m['specificity'], m['threshold']))


def compare(args):
    if not 0 <= args.specificity <= 1:
        raise ValueError('specificity 必須在 [0,1]。')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在，請指定另一個 --out。')
    parts, _ = base.load_splits(args.splits)
    expected = parts['val'].set_index('image_path').sort_index()
    report = {'dataset': 'validation_only', 'target_specificity': args.specificity,
              'note': 'Exploratory validation comparison; both thresholds/checkpoints were selected on validation. Specificity is a research setting, not a clinical standard.',
              'models': {}}
    for name, directory in [('baseline', args.baseline), ('candidate', args.candidate)]:
        directory = Path(directory)
        checkpoint = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        if checkpoint.get('threshold_policy') == 'not_yet_selected':
            raise ValueError(f'{name} 訓練尚未完成。')
        if checkpoint['split_manifest_sha256'] != base.sha_file(Path(args.splits)/'manifest.json'):
            raise ValueError(f'{name} 使用了不同切分，不能直接比較。')
        frame = pd.read_csv(directory/'validation_predictions.csv', dtype={'patient_id': str}, keep_default_na=False, encoding='utf-8-sig')
        if frame.image_path.duplicated().any() or set(frame.image_path) != set(expected.index):
            raise ValueError(f'{name} 驗證影像不一致。')
        frame = frame.set_index('image_path').sort_index()
        for field in ['label', 'file_sha256', 'pixel_sha256', 'patient_id']:
            if not frame[field].astype(str).equals(expected[field].astype(str)):
                raise ValueError(f'{name} 的 {field} 與固定驗證集不符。')
        scores = frame.score.to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('驗證分數格式錯誤。')
        fixed = base.metrics(frame.label, scores, checkpoint['threshold'])
        report['models'][name] = {'epoch': checkpoint['epoch'], 'architecture': checkpoint['architecture'],
            'checkpoint_sha256': base.sha_file(directory/'model.pt'),
            'validation_predictions_sha256': base.sha_file(directory/'validation_predictions.csv'),
            'at_saved_threshold': fixed,
            'at_validation_specificity_target': operating_point(frame.label, scores, args.specificity)}
    left, right = report['models']['baseline'], report['models']['candidate']
    report['candidate_minus_baseline'] = {
        'auroc': right['at_saved_threshold']['auroc'] - left['at_saved_threshold']['auroc'],
        'average_precision': right['at_saved_threshold']['average_precision'] - left['at_saved_threshold']['average_precision'],
        'sensitivity_at_specificity_target': right['at_validation_specificity_target']['sensitivity'] - left['at_validation_specificity_target']['sensitivity']}
    rows = []
    for model, values in report['models'].items():
        for policy in ['at_saved_threshold', 'at_validation_specificity_target']:
            m = values[policy]
            rows.append({'model': model, 'policy': policy, **{k: m[k] for k in ['threshold', 'accuracy', 'sensitivity', 'specificity', 'auroc', 'average_precision']}, **m['confusion_matrix']})
    out.mkdir(parents=True)
    base.json_save(out/'comparison.json', report)
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'完成：{out}；僅讀取驗證結果，沒有改動模型門檻。')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare-masks', help='Extract/audit masks for the original included train/val rows only')
    a.add_argument('--source-zip', required=True)
    a.add_argument('--data-root', default='data'); a.add_argument('--splits', default='splits_dfu2')
    a.add_argument('--out', default='masks_dfu2'); a.set_defaults(func=prepare_masks)
    a = sub.add_parser('train', help='Train referral classification + auxiliary wound segmentation')
    a.add_argument('--data-root', default='data'); a.add_argument('--splits', default='splits_dfu2')
    a.add_argument('--masks', default='masks_dfu2'); a.add_argument('--out', default='runs/b0_mtl_v1')
    a.add_argument('--image-size', type=int, default=224); a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--head-epochs', type=int, default=5); a.add_argument('--finetune-epochs', type=int, default=30)
    a.add_argument('--head-lr', type=float, default=1e-3); a.add_argument('--finetune-lr', type=float, default=1e-5)
    a.add_argument('--seg-weight', type=float, default=.5); a.add_argument('--patience', type=int, default=5)
    a.add_argument('--threshold-policy', choices=['max-f1', 'sensitivity'], default='max-f1')
    a.add_argument('--target-sensitivity', type=float, default=.9); a.add_argument('--class-weight', action='store_true')
    a.add_argument('--random-init', action='store_true', help='Software test only')
    a.add_argument('--workers', type=int, default=0); a.add_argument('--seed', type=int, default=42)
    a.add_argument('--threads', type=int, default=4); a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a.set_defaults(func=train)
    a = sub.add_parser('evaluate', help='Image-only evaluation; no test masks needed; exploratory on previously inspected test data')
    a.add_argument('--checkpoint', default='runs/b0_mtl_v1/model.pt'); a.add_argument('--splits', default='splits_dfu2')
    a.add_argument('--data-root', default='data'); a.add_argument('--out'); a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--bootstrap', type=int, default=500); a.add_argument('--seed', type=int, default=42)
    a.add_argument('--threads', type=int, default=4); a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a.set_defaults(func=evaluate)
    a = sub.add_parser('predict', help='Predict one new RGB image without a ground-truth mask')
    a.add_argument('--checkpoint', default='runs/b0_mtl_v1/model.pt'); a.add_argument('--image', required=True)
    a.add_argument('--out'); a.add_argument('--threads', type=int, default=4)
    a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto'); a.set_defaults(func=predict)
    a = sub.add_parser('compare', help='Compare matched validation images; does not read test predictions or change checkpoints')
    a.add_argument('--baseline', default='runs/b0_dfu2_v1'); a.add_argument('--candidate', default='runs/b0_mtl_v1')
    a.add_argument('--splits', default='splits_dfu2'); a.add_argument('--specificity', type=float, default=.8)
    a.add_argument('--out', default='comparisons/b0_mtl_v1'); a.set_defaults(func=compare)
    return p


if __name__ == '__main__':
    try:
        args = parser().parse_args()
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}')
