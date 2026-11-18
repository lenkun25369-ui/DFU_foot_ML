"""Local, supervised EfficientNet-B0 referral research pipeline.

Commands: prepare, train, evaluate, predict. See README_zh_TW.md.
Only image tensors enter the model; IDs and hashes are used for splitting/auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)


def json_save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(source):
    with Image.open(source) as im:
        return ImageOps.exif_transpose(im).convert('RGB').copy()


def table_hash(frame):
    columns = ['image_path', 'patient_id', 'label', 'file_sha256', 'pixel_sha256']
    records = frame[columns].astype(str).sort_values(['patient_id', 'image_path']).to_dict('records')
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def summary(frame):
    known = frame.patient_id.astype(str).str.strip().ne('').all()
    return {'images': len(frame), 'patients': int(frame.patient_id.nunique()) if known else None,
            'label_0': int((frame.label == 0).sum()), 'label_1': int((frame.label == 1).sum()),
            'patients_with_label_1': int(frame.loc[frame.label == 1, 'patient_id'].nunique()) if known else None}


def audit(csv_path, data_root):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'image_path', 'patient_id', 'label'}
    if not required <= set(frame.columns):
        raise ValueError(f'CSV 缺少欄位：{sorted(required - set(frame.columns))}')
    if len(frame) == 0:
        raise ValueError('CSV 沒有資料。')
    frame = frame.copy()
    frame['patient_id'] = frame.patient_id.str.strip()
    if frame.patient_id.eq('').any():
        raise ValueError('patient_id 不可空白。不可用照片流水號假裝不同病人。')
    labels = pd.to_numeric(frame.label, errors='coerce')
    if not labels.isin([0, 1]).all():
        raise ValueError('label 只接受 0 或 1；1 = 醫師標記需轉診。')
    frame['label'] = labels.astype(int)
    if frame.label.nunique() != 2:
        raise ValueError('整份資料必須同時有 label 0 和 label 1。')
    root = Path(data_root).resolve()
    paths, files, pixels, widths, heights = [], [], [], [], []
    seen_paths, seen_pixels = {}, {}
    for index, row in frame.iterrows():
        supplied = str(row.image_path).strip().replace('\\', '/')
        if not supplied:
            raise ValueError(f'CSV 第 {index + 2} 列的 image_path 空白。')
        path = (root / supplied).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'照片需放在 --data-root 內：{path}')
        if not path.is_file():
            raise FileNotFoundError(f'找不到照片（CSV 第 {index + 2} 列）：{path}')
        if str(path) in seen_paths:
            raise ValueError(f'CSV 重複指向同一張照片：第 {seen_paths[str(path)]} 與 {index + 2} 列。')
        im = read_rgb(path)
        pixel = hashlib.sha256(f'{im.size}'.encode() + im.tobytes()).hexdigest()
        if pixel in seen_pixels:
            raise ValueError(f'發現完全相同的影像像素：第 {seen_pixels[pixel]} 與 {index + 2} 列。請確認標籤並保留一份後重跑。')
        seen_pixels[pixel] = index + 2
        seen_paths[str(path)] = index + 2
        paths.append(path.relative_to(root).as_posix())
        files.append(sha_file(path))
        pixels.append(pixel)
        widths.append(im.width)
        heights.append(im.height)
    frame['image_path'] = paths
    frame['file_sha256'], frame['pixel_sha256'] = files, pixels
    frame['width'], frame['height'] = widths, heights
    return frame


def choose_split(frame, val_fraction=.15, test_fraction=.15, seed=42):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError('驗證與測試比例均需大於 0，總和需小於 1。')
    if frame.patient_id.nunique() < 6:
        raise ValueError('少於 6 位獨立病人，無法用此起始流程建立三組；請先重新設計驗證。')
    y, groups = frame.label.to_numpy(), frame.patient_id.to_numpy()
    best, best_score = None, float('inf')
    # Selection uses only labels/counts and patient IDs; no model results or image features.
    outer = GroupShuffleSplit(n_splits=400, test_size=test_fraction, random_state=seed)
    for attempt, (dev, test) in enumerate(outer.split(frame, y, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction / (1 - test_fraction),
                                  random_state=seed + attempt + 1)
        train_local, val_local = next(inner.split(frame.iloc[dev], y[dev], groups[dev]))
        indices = [dev[train_local], dev[val_local], test]
        if any(len(np.unique(y[idx])) < 2 for idx in indices):
            continue
        desired = [1 - val_fraction - test_fraction, val_fraction, test_fraction]
        score = sum(abs(len(idx) / len(frame) - frac) + abs(y[idx].mean() - y.mean())
                    for idx, frac in zip(indices, desired))
        if score < best_score:
            best, best_score = indices, score
    if best is None:
        raise ValueError('找不到每組都有兩類、且病人不重疊的切分；請檢查獨立病人數與類別分布。')
    return {name: frame.iloc[idx].reset_index(drop=True) for name, idx in zip(['train', 'val', 'test'], best)}


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在。為保留固定測試集，請使用新的 --out，或自行確認後處理舊檔。')
    frame = audit(args.csv, args.data_root)
    splits = choose_split(frame, args.val_fraction, args.test_fraction, args.seed)
    out.mkdir(parents=True)
    manifest = {'schema_version': 1, 'seed': args.seed, 'label_definition': args.label_definition,
                'patient_separation_verified': True,
                'data_root_at_preparation': str(Path(args.data_root).resolve()),
                'source_csv_sha256': sha_file(args.csv), 'splits': {}}
    for name, part in splits.items():
        part.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
        manifest['splits'][name] = {**summary(part), 'table_sha256': table_hash(part)}
    json_save(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('完成。這是依病人切分；實際照片數可能不是精確 70/15/15。')
    print('完全重複像素已檢查；旋轉、壓縮後的近似重複與病人 ID 正確性仍需人工確認。')


def load_splits(directory, expected_hash=None):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if expected_hash and sha_file(manifest_path) != expected_hash:
        raise ValueError('切分 manifest 已改變，不能使用這個模型評估新的測試切分。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    parts = {}
    for name in ['train', 'val', 'test']:
        frame = pd.read_csv(directory / f'{name}.csv', dtype={'patient_id': str, 'image_path': str},
                            keep_default_na=False, encoding='utf-8-sig')
        if not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2:
            raise ValueError(f'{name} 必須包含 0 和 1 兩類。')
        if table_hash(frame) != manifest['splits'][name]['table_sha256']:
            raise ValueError(f'{name}.csv 與 manifest 不符；不可手動改標籤或切分後沿用舊模型。')
        parts[name] = frame
    verified = manifest.get('patient_separation_verified', True)
    if not isinstance(verified, bool):
        raise ValueError('patient_separation_verified 必須為布林值。')
    if verified and any(f.patient_id.astype(str).str.strip().eq('').any() for f in parts.values()):
        raise ValueError('宣稱病人分組驗證，但 patient_id 有空白。')
    if not verified and any(f.patient_id.astype(str).str.strip().ne('').any() for f in parts.values()):
        raise ValueError('未驗證病人模式不得混入部分 patient_id；請完成對應後重新匯入。')
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for key in (['patient_id'] if verified else []) + ['image_path', 'pixel_sha256']:
            if set(parts[left][key]) & set(parts[right][key]):
                raise ValueError(f'{left}/{right} 發現 {key} 重疊。')
    return parts, manifest


def check_files(frame, root):
    root = Path(root).resolve()
    for row in frame.itertuples():
        path = (root / row.image_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha_file(path) != row.file_sha256:
            raise ValueError(f'照片遺失或自 prepare 後已變更：{path}。請重新建立研究版本。')


class Letterbox:
    """Full field of view, aspect ratio preserved; training and inference share this."""
    def __init__(self, size=224):
        self.size = int(size)

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BICUBIC,
                            color=(124, 116, 104), centering=(.5, .5))


def image_transform(size=224, training=False):
    from torchvision import transforms as T
    ops = [Letterbox(size)]
    if training:
        # Preserve wound/periwound context. No random crop or aggressive colour jitter.
        ops += [T.RandomHorizontalFlip(p=.5)]
    ops += [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    return T.Compose(ops)


class ImageDataset:
    def __init__(self, frame, root, size, training=False):
        self.frame, self.root = frame.reset_index(drop=True), Path(root)
        self.transform = image_transform(size, training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        import torch
        row = self.frame.iloc[index]
        x = self.transform(read_rgb(self.root / row.image_path))
        return x, torch.tensor(float(row.label), dtype=torch.float32)


def create_model(pretrained=False):
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def device_for(name='auto'):
    import torch
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定 CUDA，但 torch.cuda.is_available() 為 False。先執行 check_env.py。')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else ('cpu' if name == 'auto' else name))


def loader_for(frame, root, size, batch, workers=0, training=False, seed=42):
    import torch
    return torch.utils.data.DataLoader(ImageDataset(frame, root, size, training), batch_size=batch,
        shuffle=training, num_workers=workers, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed))


def get_scores(model, loader, device):
    import torch
    model.eval()
    output = []
    with torch.inference_mode():
        for x, _ in loader:
            output.extend(torch.sigmoid(model(x.to(device)).flatten()).cpu().tolist())
    return np.asarray(output, dtype=float)


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('預測分數數量錯誤或包含非有限值。')
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else None
    both = len(np.unique(y)) == 2
    return {'n_images': len(y), 'threshold': float(threshold),
            'accuracy': float((prediction == y).mean()),
            'sensitivity': ratio(tp, tp + fn), 'specificity': ratio(tn, tn + fp),
            'precision': ratio(tp, tp + fp), 'npv': ratio(tn, tn + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn),
            'auroc': float(roc_auc_score(y, scores)) if both else None,
            'average_precision': float(average_precision_score(y, scores)) if both else None,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}


def select_threshold(y, scores, policy, target=.90):
    # Threshold is chosen exclusively using validation predictions.
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    assessed = [metrics(y, scores, t) for t in candidates]
    if policy == 'sensitivity':
        if not 0 < target <= 1:
            raise ValueError('target-sensitivity 必須在 (0,1]。')
        possible = [m for m in assessed if m['sensitivity'] >= target]
        best = max(possible, key=lambda m: (m['specificity'], m['threshold']))
    else:
        best = max(assessed, key=lambda m: (m['f1'] or 0, m['specificity'], m['threshold']))
    return float(best['threshold'])


def seed_everything(seed, threads=4):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device='cpu'):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'efficientnet_b0' or checkpoint.get('schema_version') != 1:
        raise ValueError('不支援的模型檔；請使用本套程式 train 產生的 checkpoint。')
    model = create_model(False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def train(args):
    import torch
    from torch import nn
    import torchvision
    import sklearn
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在；換一個 --out 可保留原本實驗。')
    if args.head_epochs < 1 or args.finetune_epochs < 0 or args.patience < 1:
        raise ValueError('head-epochs >= 1、finetune-epochs >= 0、patience >= 1。')
    seed_everything(args.seed, args.threads)
    parts, manifest = load_splits(args.splits)
    if not manifest.get('patient_separation_verified', True):
        print('探索性影像切分：沒有病人 ID，無法確認新病人泛化能力。', flush=True)
    for key in ['train', 'val']:
        check_files(parts[key], args.data_root)
    device = device_for(args.device)
    print(f'Device: {device}; train/val only. Test images are not loaded during training.', flush=True)
    # Explicit --random-init is only for checking that the software runs offline.
    model = create_model(not args.random_init).to(device)
    out.mkdir(parents=True)
    config = vars(args).copy()
    config.pop('func', None)
    config['versions'] = {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                          'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}
    config['split_summary'] = manifest['splits']
    config['patient_separation_verified'] = manifest.get('patient_separation_verified', True)
    json_save(out / 'config.json', config)
    train_loader = loader_for(parts['train'], args.data_root, args.image_size, args.batch_size,
                              args.workers, True, args.seed)
    val_loader = loader_for(parts['val'], args.data_root, args.image_size, args.batch_size, args.workers)
    positive = int(parts['train'].label.sum())
    negative = len(parts['train']) - positive
    pos_weight = negative / positive if args.class_weight else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history, best_ap, global_epoch = [], -1.0, 0
    checkpoint_path = out / 'model.pt'
    for stage, epochs, lr in [('head', args.head_epochs, args.head_lr),
                              ('finetune', args.finetune_epochs, args.finetune_lr)]:
        if epochs == 0:
            continue
        for parameter in model.features.parameters():
            parameter.requires_grad = stage == 'finetune'
        # Fine-tune from the best frozen-head checkpoint, not merely the last epoch.
        if stage == 'finetune':
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['state_dict'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=lr, weight_decay=1e-4)
        bad_epochs, stage_best = 0, -1.0
        for _ in range(epochs):
            global_epoch += 1
            model.train()
            # Frozen backbone must also stop updating BatchNorm running statistics.
            if stage == 'head':
                model.features.eval()
            total_loss, total_seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).flatten()
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss 出現 NaN/Inf；停止這次訓練並檢查資料與學習率。')
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(y)
                total_seen += len(y)
            scores = get_scores(model, val_loader, device)
            vm = metrics(parts['val'].label.to_numpy(), scores, .5)
            y_val = torch.tensor(parts['val'].label.to_numpy(), dtype=torch.float32)
            # Report unweighted validation BCE on probabilities, independent of train weighting.
            val_bce = float(nn.functional.binary_cross_entropy(torch.tensor(scores, dtype=torch.float32), y_val))
            row = {'epoch': global_epoch, 'stage': stage, 'train_loss': total_loss / total_seen,
                   'val_loss': val_bce, 'val_auroc': vm['auroc'], 'val_average_precision': vm['average_precision']}
            history.append(row)
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(json.dumps(row), flush=True)
            ap = vm['average_precision']
            if ap > best_ap + 1e-8:
                best_ap = ap
                checkpoint = {'schema_version': 1, 'architecture': 'efficientnet_b0',
                    'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'image_size': args.image_size, 'preprocessing': 'exif_rgb_letterbox_imagenet_v1',
                    'threshold': .5, 'threshold_policy': 'not_yet_selected', 'epoch': global_epoch,
                    'split_manifest_sha256': sha_file(Path(args.splits) / 'manifest.json'),
                    'label_definition': manifest['label_definition'], 'positive_label': 1,
                    'patient_separation_verified': manifest.get('patient_separation_verified', True),
                    'initialization': 'random_test_only' if args.random_init else 'imagenet1k_v1',
                    'val_average_precision': best_ap, 'seed': args.seed, 'pos_weight': pos_weight,
                    'versions': config['versions']}
                torch.save(checkpoint, checkpoint_path)
            if ap > stage_best + 1e-8:
                stage_best, bad_epochs = ap, 0
            else:
                bad_epochs += 1
            if stage == 'finetune' and bad_epochs >= args.patience:
                print('Early stopping: validation average precision stopped improving.', flush=True)
                break
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    scores = get_scores(model, val_loader, device)
    threshold = select_threshold(parts['val'].label.to_numpy(), scores, args.threshold_policy, args.target_sensitivity)
    checkpoint.update({'threshold': threshold, 'threshold_policy': args.threshold_policy,
                       'target_sensitivity': args.target_sensitivity if args.threshold_policy == 'sensitivity' else None})
    torch.save(checkpoint, checkpoint_path)
    json_save(out / 'validation_metrics.json', {'at_0_5': metrics(parts['val'].label, scores, .5),
               'at_selected_threshold': metrics(parts['val'].label, scores, threshold),
               'note': 'Validation was used for model/threshold selection; these are not unbiased test estimates.'})
    predictions = parts['val'].copy()
    predictions['score'], predictions['prediction'] = scores, (scores >= threshold).astype(int)
    predictions.to_csv(out / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    plot_history(pd.DataFrame(history), out / 'training_curves.png')
    print(f'完成：{checkpoint_path}；驗證集門檻={threshold:.6f}。尚未使用測試集計算成績。')


def plot_history(frame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label='Train BCE (weighted if enabled)')
    axes[0].plot(frame.epoch, frame.val_loss, label='Validation BCE (unweighted)')
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[0].legend(fontsize=8)
    axes[1].plot(frame.epoch, frame.val_average_precision, label='Validation AP')
    axes[1].plot(frame.epoch, frame.val_auroc, label='Validation AUROC')
    axes[1].set(xlabel='Epoch', ylabel='Score', ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cluster_bootstrap(frame, scores, threshold, repetitions=500, seed=42):
    rng = np.random.default_rng(seed)
    ids = frame.patient_id.unique()
    indices = {pid: np.flatnonzero(frame.patient_id.to_numpy() == pid) for pid in ids}
    fields = ['auroc', 'average_precision', 'sensitivity', 'specificity', 'precision', 'npv', 'f1']
    values = {field: [] for field in fields}
    y = frame.label.to_numpy()
    for _ in range(repetitions):
        sample = np.concatenate([indices[pid] for pid in rng.choice(ids, size=len(ids), replace=True)])
        if len(np.unique(y[sample])) < 2:
            continue
        measured = metrics(y[sample], scores[sample], threshold)
        for field in fields:
            if measured[field] is not None:
                values[field].append(measured[field])
    return {field: {'lower': float(np.percentile(v, 2.5)) if v else None,
                    'upper': float(np.percentile(v, 97.5)) if v else None,
                    'valid_resamples': len(v)} for field, v in values.items()}


def evaluate(args):
    seed_everything(args.seed, args.threads)
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get('threshold_policy') == 'not_yet_selected':
        raise ValueError('這次訓練尚未完成驗證與門檻選擇，請先完成 train。')
    parts, manifest = load_splits(args.splits, checkpoint['split_manifest_sha256'])
    frame = parts['test']
    check_files(frame, args.data_root)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / 'test'
    if out.exists():
        raise FileExistsError(f'{out} 已存在。測試集不應反覆用於調參；如需重現同一模型請明確指定新的 --out。')
    scores = get_scores(model, loader_for(frame, args.data_root, checkpoint['image_size'], args.batch_size), device)
    threshold = checkpoint['threshold']
    out.mkdir(parents=True)
    verified = manifest.get('patient_separation_verified', True)
    report = {'unit': 'image', 'patients': int(frame.patient_id.nunique()) if verified else None,
              'patient_separation_verified': verified,
              'checkpoint_sha256': sha_file(args.checkpoint), 'label_definition': checkpoint['label_definition'],
              'at_selected_threshold': metrics(frame.label, scores, threshold),
              'at_0_5': metrics(frame.label, scores, .5),
              'ci95_patient_cluster_bootstrap': cluster_bootstrap(frame, scores, threshold, args.bootstrap, args.seed) if verified else None,
              'ci_note': ('Image-level estimates; whole patients resampled. Small cohorts yield unstable intervals.'
                          if verified else 'No patient IDs: no patient-cluster CI; split independence and new-patient performance are unverified.'),
              'average_precision_note': 'AP is average precision, not trapezoidal PR-AUC.'}
    json_save(out / 'metrics.json', report)
    predictions = frame.copy()
    predictions['score'], predictions['prediction'] = scores, (scores >= threshold).astype(int)
    predictions.to_csv(out / 'predictions.csv', index=False, encoding='utf-8-sig')
    predictions.loc[(predictions.label == 1) & (predictions.prediction == 0)].to_csv(
        out / 'false_negatives.csv', index=False, encoding='utf-8-sig')
    plot_evaluation(frame.label, scores, threshold, out / 'evaluation.png')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def plot_evaluation(y, scores, threshold, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fpr, tpr, _ = roc_curve(y, scores)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='Sensitivity', title='ROC')
    precision, recall, _ = precision_recall_curve(y, scores)
    axes[1].plot(recall, precision)
    axes[1].axhline(float(np.mean(y)), ls='--', color='gray')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall')
    ConfusionMatrixDisplay(confusion_matrix(y, scores >= threshold, labels=[0, 1]),
                           display_labels=['No referral', 'Referral']).plot(ax=axes[2], colorbar=False)
    axes[2].set_title(f'Validation-selected threshold: {threshold:.3f}')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def predict_image(model, checkpoint, source, device):
    import torch
    if checkpoint.get('threshold_policy') == 'not_yet_selected':
        raise ValueError('這次訓練尚未完成驗證與門檻選擇，請先完成 train。')
    x = image_transform(checkpoint['image_size'])(read_rgb(source)).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = float(torch.sigmoid(model(x).flatten())[0].cpu())
    return {'score': score, 'threshold': checkpoint['threshold'],
            'prediction': int(score >= checkpoint['threshold']),
            'label_definition': checkpoint['label_definition'],
            'patient_separation_verified': checkpoint.get('patient_separation_verified', True),
            'note': 'Research prediction; score is not a calibrated clinical probability.'}


def predict(args):
    import torch
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    result = predict_image(model, checkpoint, args.image, device)
    if args.out:
        if Path(args.out).exists():
            raise FileExistsError(f'{args.out} 已存在。')
        json_save(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare', help='Audit image/label pairing and split by patient')
    a.add_argument('--csv', default='data/labels.csv')
    a.add_argument('--data-root', default='data')
    a.add_argument('--out', default='splits')
    a.add_argument('--val-fraction', type=float, default=.15)
    a.add_argument('--test-fraction', type=float, default=.15)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--label-definition', default='醫師標記之轉診需求：0=不需轉診；1=需轉診；未定義為緊急轉診')
    a.set_defaults(func=prepare)
    a = sub.add_parser('train', help='Train on train split; select checkpoint/threshold on validation only')
    a.add_argument('--splits', default='splits')
    a.add_argument('--data-root', default='data')
    a.add_argument('--out', default='runs/b0_v1')
    a.add_argument('--image-size', type=int, default=224)
    a.add_argument('--batch-size', type=int, default=16)
    a.add_argument('--head-epochs', type=int, default=5)
    a.add_argument('--finetune-epochs', type=int, default=30)
    a.add_argument('--head-lr', type=float, default=1e-3)
    a.add_argument('--finetune-lr', type=float, default=1e-5)
    a.add_argument('--patience', type=int, default=5)
    a.add_argument('--class-weight', action='store_true', help='Optional BCE positive weight = train negatives / positives')
    a.add_argument('--threshold-policy', choices=['max-f1', 'sensitivity'], default='max-f1')
    a.add_argument('--target-sensitivity', type=float, default=.90,
                   help='Exploratory validation target ONLY; not a clinical safety guarantee')
    a.add_argument('--random-init', action='store_true', help='Offline software test only; normal research uses pretrained weights')
    a.add_argument('--workers', type=int, default=0)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--threads', type=int, default=4)
    a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a.set_defaults(func=train)
    a = sub.add_parser('evaluate', help='Evaluate the fixed checkpoint on the reserved test set')
    a.add_argument('--checkpoint', default='runs/b0_v1/model.pt')
    a.add_argument('--splits', default='splits')
    a.add_argument('--data-root', default='data')
    a.add_argument('--out')
    a.add_argument('--batch-size', type=int, default=16)
    a.add_argument('--bootstrap', type=int, default=500)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--threads', type=int, default=4)
    a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a.set_defaults(func=evaluate)
    a = sub.add_parser('predict', help='Predict one new image; no label required')
    a.add_argument('--checkpoint', default='runs/b0_v1/model.pt')
    a.add_argument('--image', required=True)
    a.add_argument('--out')
    a.add_argument('--threads', type=int, default=4)
    a.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a.set_defaults(func=predict)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}')
