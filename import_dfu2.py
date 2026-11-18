"""Import the supplied DFU2 Excel and image ZIP; preserve source split membership.
Only DFU2/{train,validation,test}/images are selected. Masks are not model inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook
from dfu import read_rgb, sha_file, json_save, summary, table_hash, load_splits

SPLITS = ('train', 'val', 'test')
FOLDERS = {'train': 'train', 'val': 'validation', 'test': 'test'}
LABEL_DEFINITION = '醫師標記之轉診需求：0=不需轉診；1=需轉診；未定義為緊急轉診'


def normalize_header(value):
    return str(value or '').strip().removeprefix('\ufeff').removeprefix('ï»¿').strip()


def normalize_split(value):
    value = str(value).strip().lower()
    value = {'validation': 'val', 'valid': 'val'}.get(value, value)
    if value not in SPLITS:
        raise ValueError(f'未知的 split：{value}')
    return value


def canonical_key(path):
    """Ignore a garbled outer folder, but match the complete DFU2 images suffix."""
    parts = PurePosixPath(str(path).replace('\\', '/')).parts
    if '..' in parts:
        raise ValueError('路徑不能包含 ..')
    hits = []
    for i, part in enumerate(parts):
        tail = parts[i:]
        if part == 'DFU2' and len(tail) == 4 and tail[2] == 'images':
            hits.append((normalize_split(tail[1]), tail[3]))
    if len(hits) > 1:
        raise ValueError('DFU2 路徑不明確。')
    return hits[0] if hits else None


def read_labels(path, sheet=None):
    book = load_workbook(path, read_only=True, data_only=True)
    required = {'split', 'image_name', 'zip_path', 'label'}
    candidates = []
    try:
        for ws in book:
            if sheet and ws.title != sheet:
                continue
            iterator = ws.iter_rows(values_only=True)
            headers = [normalize_header(v) for v in next(iterator, ())]
            if required <= set(headers):
                if len(headers) != len(set(headers)):
                    raise ValueError('Excel 欄名重複。')
                candidates.append((ws.title, headers, list(iterator)))
        if len(candidates) != 1:
            raise ValueError('應有一個包含 split/image_name/zip_path/label 的工作表；多個時請指定 --sheet。')
        title, headers, values = candidates[0]
        rows = []
        for number, values_row in enumerate(values, 2):
            if all(v is None for v in values_row):
                continue
            row = dict(zip(headers, values_row))
            split = normalize_split(row['split'])
            name = str(row['image_name'] or '').strip()
            if not name or PurePosixPath(name).name != name or '\\' in name or ':' in name or name in {'.', '..'}:
                raise ValueError(f'Excel 第 {number} 列 image_name 不是單純檔名。')
            if canonical_key(row['zip_path']) != (split, name):
                raise ValueError(f'Excel 第 {number} 列 zip_path 與 split/image_name 不一致。')
            raw = row['label']
            text = str(raw).strip() if raw is not None else ''
            if text in {'', '?'}:
                label = None
            else:
                try:
                    numeric = float(text)
                except ValueError:
                    numeric = float('nan')
                if numeric not in (0.0, 1.0):
                    raise ValueError(f'Excel 第 {number} 列 label 只接受 0、1 或 ?；收到 {raw!r}')
                label = int(numeric)
            rows.append({'source_row': number, 'split': split, 'image_name': name,
                         'source_zip_path': str(row['zip_path']), 'label_raw': text,
                         'label': label, 'patient_id': ''})
    finally:
        book.close()
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(['split', 'image_name']).any():
        raise ValueError('標籤表空白，或同一 split/image_name 出現多列。')
    return frame, title


def attach_images(frame, data_root, source_zip=None, source_dir=None):
    root = Path(data_root).resolve()
    source = ZipFile(source_zip) if source_zip else None
    index = {}
    try:
        entries = source.namelist() if source else (str(p) for p in Path(source_dir).resolve().rglob('*') if p.is_file())
        for entry in entries:
            # Skip masks, DFU1/3 and thermal images completely.
            key = canonical_key(entry)
            if key:
                if key in index:
                    raise ValueError(f'影像來源有多個相同的 DFU2 路徑：{key}')
                index[key] = entry
        result = frame.copy()
        for i, row in result.iterrows():
            key = (row['split'], row['image_name'])
            if key not in index:
                raise FileNotFoundError(f'找不到 Excel 第 {row.source_row} 列的影像：{key}')
            relative = Path('DFU2') / FOLDERS[row['split']] / 'images' / row['image_name']
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError('輸出影像路徑超出 data-root。')
            if source:
                data = source.read(index[key])
            else:
                data = Path(index[key]).read_bytes()
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise FileExistsError(f'既有照片與來源不同，不覆寫：{target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            im = read_rgb(target)
            result.loc[i, 'image_path'] = relative.as_posix()
            result.loc[i, 'file_sha256'] = hashlib.sha256(data).hexdigest()
            result.loc[i, 'pixel_sha256'] = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            result.loc[i, 'width'], result.loc[i, 'height'] = im.size
        result[['width', 'height']] = result[['width', 'height']].astype(int)
        return result
    finally:
        if source:
            source.close()


def apply_patient_map(frame, path):
    if not path:
        return frame
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'split', 'image_name', 'patient_id'}
    if not required <= set(mapping):
        raise ValueError('病人對應表需有 split,image_name,patient_id。')
    mapping['split'] = mapping['split'].map(normalize_split)
    mapping['image_name'] = mapping.image_name.str.strip()
    mapping['patient_id'] = mapping.patient_id.str.strip()
    if mapping.duplicated(['split', 'image_name']).any():
        raise ValueError('病人對應表含重複 split/image_name。')
    result = frame.drop(columns='patient_id').merge(mapping[list(required)], on=['split', 'image_name'], how='left', validate='one_to_one')
    result['patient_id'] = result.patient_id.fillna('')
    known = result[result.label.notna()]
    if known.patient_id.eq('').any():
        raise ValueError('所有已標記 0/1 的照片都需有真實研究用 patient_id；不可只填一部分。')
    # Check original labeled rows before deduplication to expose patient overlap.
    overlap = known.groupby('patient_id')['split'].nunique().gt(1)
    if overlap.any():
        raise ValueError('病人對應表顯示同一病人跨組。需重新設計病人分組；不可宣稱此既有分組為病人獨立驗證。')
    return result


def resolve_duplicates(frame):
    """Keep test > val > train, then lexical filename; never use model scores."""
    records, keep = [], []
    for pixel, group in frame.groupby('pixel_sha256', sort=False):
        known = group[group.label.notna()]
        if known.label.nunique() > 1:
            rows = ','.join(map(str, group.source_row.tolist()))
            raise ValueError(f'完全相同影像存在 0/1 衝突，請由標註者確認；Excel 列 {rows}')
        if known.empty:
            continue
        order = known.assign(priority=known['split'].map({'test': 0, 'val': 1, 'train': 2}))
        order = order.sort_values(['priority', 'image_name', 'source_row'])
        selected = order.iloc[0]
        keep.append(selected.name)
        for index in order.index[1:]:
            removed = frame.loc[index].to_dict()
            removed.update({'reason': 'exact_duplicate', 'kept_image_path': selected.image_path,
                            'kept_split': selected['split']})
            records.append(removed)
    cleaned = frame.loc[keep].copy().sort_values(['split', 'image_name']).reset_index(drop=True)
    cleaned['label'] = cleaned.label.astype(int)
    excluded = pd.DataFrame(records, columns=[*frame.columns, 'reason', 'kept_image_path', 'kept_split'])
    return cleaned, excluded


def split_counts(frame):
    result = {}
    for split in SPLITS:
        part = frame[frame['split'].eq(split)]
        result[split] = {'images': len(part), 'label_0': int(part.label.eq(0).sum()),
                         'label_1': int(part.label.eq(1).sum()), 'unknown': int(part.label.isna().sum())}
    return result


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} 已存在；使用新的 --out 保留資料版本。')
    if args.source_dir and not Path(args.source_dir).is_dir():
        raise FileNotFoundError(args.source_dir)
    frame, sheet = read_labels(args.excel, args.sheet)
    frame = apply_patient_map(frame, args.patient_map)
    frame = attach_images(frame, args.data_root, args.source_zip, args.source_dir)
    cleaned, excluded_duplicates = resolve_duplicates(frame)
    unknown = frame[frame.label.isna()].copy()
    unknown['reason'] = 'unknown_label'
    grouped = [g for _, g in frame.groupby('pixel_sha256') if len(g) > 1]
    duplicate_report = [{'pixel_sha256': g.pixel_sha256.iloc[0], 'cross_split': bool(g['split'].nunique() > 1),
                         'members': g[['source_row', 'split', 'image_name', 'label_raw']].to_dict('records')} for g in grouped]
    parts = {s: cleaned[cleaned['split'].eq(s)].reset_index(drop=True) for s in SPLITS}
    if any(p.label.nunique() != 2 for p in parts.values()):
        raise ValueError('排除未知標籤與重複照片後，每組仍需同時有 0 和 1。')
    report = {'source_excel_sha256': sha_file(args.excel), 'sheet': sheet,
              'source_counts': split_counts(frame), 'included_counts': split_counts(cleaned),
              'unknown_labels_excluded': len(unknown), 'known_duplicate_copies_excluded': len(excluded_duplicates),
              'exact_duplicate_groups_all_images': len(grouped),
              'cross_split_duplicate_groups_all_images': sum(g['split'].nunique() > 1 for g in grouped),
              'patient_separation_verified': bool(args.patient_map),
              'deduplication_rule': 'known labels only; test > val > train; then lexical image_name; conflicts stop import',
              'label_definition': args.label_definition,
              'notes': ['Original split membership retained; duplicate copies excluded, not reassigned.',
                        'Unknown labels are excluded, never converted to 0 or inferred from masks.',
                        'No patient map means no verified patient separation or patient-cluster confidence interval.',
                        'Pixel hashing does not detect all near duplicates; manual review still needed.']}
    manifest = {'schema_version': 1, 'seed': None, 'label_definition': args.label_definition,
                'patient_separation_verified': bool(args.patient_map), 'split_origin': 'DFU2 Excel predefined, exact duplicates removed',
                'data_root_at_preparation': str(Path(args.data_root).resolve()),
                'source_excel_sha256': report['source_excel_sha256'], 'source_sheet': sheet,
                'patient_map_sha256': sha_file(args.patient_map) if args.patient_map else None,
                'splits': {s: {**summary(p), 'table_sha256': table_hash(p)} for s, p in parts.items()}}
    out.mkdir(parents=True)
    for name, part in parts.items():
        part.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    unknown.to_csv(out / 'excluded_unknown.csv', index=False, encoding='utf-8-sig')
    excluded_duplicates.to_csv(out / 'excluded_duplicates.csv', index=False, encoding='utf-8-sig')
    frame[['split', 'image_name', 'patient_id']].to_csv(out / 'patient_map_template.csv', index=False, encoding='utf-8-sig')
    json_save(out / 'duplicate_groups.json', duplicate_report)
    json_save(out / 'audit_report.json', report)
    json_save(out / 'manifest.json', manifest)
    load_splits(out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'完成：{out}；原始 Excel/ZIP 保持不變。')
    if not args.patient_map:
        print('可開始探索性影像分類。沒有 patient_id，不能宣稱新病人驗證。')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--excel', required=True)
    p.add_argument('--sheet')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-zip', help='Original ZIP containing DFU2 images')
    source.add_argument('--source-dir', help='Extracted directory containing DFU2, or DFU2 itself')
    p.add_argument('--data-root', default='data')
    p.add_argument('--out', default='splits_dfu2')
    p.add_argument('--patient-map', help='Optional CSV with split,image_name,patient_id; must be complete for labeled images')
    p.add_argument('--label-definition', default=LABEL_DEFINITION)
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f'ERROR: {exc}')
