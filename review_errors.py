"""Local review tool for the DFU referral project (Python 3.10+).

Place beside dfu.py, then run in the dfu_referral environment:
  python -m streamlit run review_errors.py --server.address 127.0.0.1 -- \
    --predictions runs/b0_dfu2_v1/test/predictions.csv --data-root data \
    --review-out reviews/b0_dfu2_v1_review.csv

Open http://localhost:8501. No model loading or retraining is required.
Review records are written only after pressing Save. The original labels,
predictions, images, checkpoint and threshold are not changed.
Photo appearance alone cannot verify clinical referral necessity. Check the
original annotation record; a disputed label requires the annotator's review.
Test-set review is exploratory; subsequent changes need independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st


GROUPS = {'FN': '漏判：標籤 1，模型 0', 'FP': '誤報：標籤 0，模型 1',
          'TP': '判對：標籤 1，模型 1', 'TN': '判對：標籤 0，模型 0', 'ALL': '全部影像'}
STATUSES = ['未檢查', '已檢查', '待醫師確認', '待核對資料來源']
LABEL_REVIEWS = ['未核對原始標註', '已核對，維持原標籤', '疑似影像與標籤配對錯誤',
                 '原始標註有疑義，請標註者確認', '缺少判讀所需的臨床資訊']
ISSUES = ['模糊', '過暗或過曝', '反光或色偏', '遮擋', '傷口範圍不清楚',
          '拍攝距離或角度差異', '文字、浮水印或背景線索', '疑似相同病人或近似照片']
REVIEW_COLUMNS = ['predictions_sha256', 'image_path', 'original_label', 'prediction',
                  'score', 'error_group', 'status', 'label_review', 'image_issues',
                  'source_note', 'patient_group_note', 'reviewer', 'notes', 'updated_at_utc']


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(path):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'image_path', 'label', 'prediction', 'score'}
    if not required <= set(frame) or frame.empty:
        raise ValueError('預測 CSV 需有 image_path、label、prediction、score，且不可為空。')
    if frame.image_path.eq('').any() or frame.image_path.duplicated().any():
        raise ValueError('預測 CSV 的影像路徑空白或重複。')
    for field in ['label', 'prediction']:
        values = pd.to_numeric(frame[field], errors='coerce')
        if not values.isin([0, 1]).all():
            raise ValueError(f'{field} 必須是 0/1。')
        frame[field] = values.astype(int)
    scores = pd.to_numeric(frame.score, errors='coerce')
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError('score 必須在 0–1 之間且不可缺失。')
    frame['score'] = scores.astype(float)
    frame['error_group'] = ['TP' if y == p == 1 else 'TN' if y == p == 0 else 'FN' if y == 1 else 'FP'
                            for y, p in zip(frame.label, frame.prediction)]
    threshold = None
    metrics_path = path.parent / ('validation_metrics.json' if path.name == 'validation_predictions.csv' else 'metrics.json')
    if metrics_path.is_file():
        report = json.loads(metrics_path.read_text(encoding='utf-8'))
        threshold = float(report['at_selected_threshold']['threshold'])
        if not 0 <= threshold <= 1 or not (frame.prediction == (frame.score >= threshold).astype(int)).all():
            raise ValueError('預測 CSV 與同目錄 metrics 的分類門檻不一致。')
    return frame, sha_file(path), threshold


def load_reviews(path, prediction_hash):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if list(frame.columns) != REVIEW_COLUMNS:
        raise ValueError('既有查核紀錄格式不符；請指定另一個 --review-out，避免覆寫其他檔案。')
    if frame.predictions_sha256.ne(prediction_hash).any():
        raise ValueError('這份查核紀錄屬於不同預測版本；請指定新的 --review-out。')
    if frame.image_path.duplicated().any():
        raise ValueError('既有查核紀錄含重複照片；請先核對。')
    return frame


def save_review(path, record, source_csv):
    path, source = Path(path).resolve(), Path(source_csv).resolve()
    if path == source or path.suffix.lower() != '.csv':
        raise ValueError('查核紀錄需另存為 CSV，不可覆寫原始預測。')
    # Re-read at save time so separately reviewed cases are retained.
    previous = load_reviews(path, record['predictions_sha256'])
    previous = previous.loc[previous.image_path.ne(record['image_path'])]
    output = pd.concat([previous, pd.DataFrame([record], columns=REVIEW_COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                         dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            output.to_csv(f, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_photo(data_root, row):
    root = Path(data_root).resolve()
    photo = (root / str(row.image_path).replace('\\', '/')).resolve()
    if not photo.is_relative_to(root):
        raise ValueError('照片路徑超出 data-root。')
    if not photo.is_file():
        raise FileNotFoundError(f'找不到照片：{photo}')
    expected = row.get('file_sha256', '')
    if expected and sha_file(photo) != expected:
        raise ValueError('照片與預測時的檔案雜湊不符，請核對來源，避免看錯版本。')
    with Image.open(photo) as im:
        return ImageOps.exif_transpose(im).convert('RGB').copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='runs/b0_dfu2_v1/test/predictions.csv')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--review-out', default='reviews/b0_dfu2_v1_review.csv')
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title='DFU 影像錯誤檢查', layout='wide')
    st.title('DFU 影像錯誤檢查')
    st.caption('逐張核對影像品質、資料來源與原始標註，將人工檢查紀錄另存。')
    try:
        frame, prediction_hash, threshold = load_predictions(args.predictions)
        reviews = load_reviews(args.review_out, prediction_hash)
        if Path(args.predictions).resolve() == Path(args.review_out).resolve():
            raise ValueError('查核紀錄路徑不能與原始預測相同。')
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    totals = frame.error_group.value_counts()
    a, b, c, d = st.columns(4)
    for column, group in zip([a, b, c, d], ['FN', 'FP', 'TP', 'TN']):
        column.metric(GROUPS[group], int(totals.get(group, 0)))
    if threshold is not None:
        st.caption(f'本次固定門檻：{threshold:.6f}；分數為模型輸出，並非已校正的臨床風險。')
    group = st.sidebar.selectbox('查看類別', list(GROUPS), format_func=lambda x: GROUPS[x])
    order = st.sidebar.selectbox('排序', ['分數由低到高', '分數由高到低', '檔名'])
    pending_only = st.sidebar.checkbox('只看尚未存檔的照片')
    selection = frame if group == 'ALL' else frame.loc[frame.error_group.eq(group)]
    if pending_only:
        selection = selection.loc[~selection.image_path.isin(reviews.image_path)]
    selection = selection.sort_values(['image_path'] if order == '檔名' else ['score', 'image_path'],
                                     ascending=order != '分數由高到低')
    st.sidebar.write(f'已存檔：{len(reviews)} / {len(frame)} 張')
    if not reviews.empty:
        st.sidebar.download_button('下載目前查核紀錄', reviews.to_csv(index=False).encode('utf-8-sig'),
                                   file_name=Path(args.review_out).name, mime='text/csv')
    st.sidebar.caption('要改變來源，重新執行時指定 --predictions；驗證與測試資料請使用不同的查核檔。')
    if selection.empty:
        st.info('目前條件下沒有照片。可切換類別或取消「只看尚未存檔」。')
        st.stop()
    display = selection.reset_index(drop=True)
    picked = st.selectbox('選擇照片', list(range(len(display))),
                         format_func=lambda i: f'{i + 1}/{len(display)} · {Path(display.iloc[i].image_path).name} · 分數 {display.iloc[i].score:.3f}',
                         key=f'photo_{group}_{order}_{pending_only}')
    row = display.iloc[picked]
    saved = reviews.loc[reviews.image_path.eq(row.image_path)]
    existing = saved.iloc[0].to_dict() if not saved.empty else {}
    left, right = st.columns([1.15, 1])
    with left:
        try:
            st.image(load_photo(args.data_root, row), caption=row.image_path, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.write(f'原始標籤：**{row.label}**　模型預測：**{row.prediction}**　分數：**{row.score:.4f}**')
        st.caption('本專案定義：1＝需轉診，0＝不需轉診。分數最低的漏判代表模型更偏向判為 0，不表示臨床最嚴重。')
        st.info('先看照片是否清楚、傷口是否可見，再核對拍攝來源及原始標註紀錄。疑似標籤問題交由標註者確認，不能只因模型判斷不同就改答案。')
        st.write('比較方式：看完漏判，可切換「判對：標籤 1，模型 1」，比較相似外觀的照片是否也會判對。')
    with right:
        st.caption('填寫後請先按儲存，再切換照片。')
        key = hashlib.sha256((prediction_hash + row.image_path).encode()).hexdigest()[:16]
        with st.form(f'review_{key}'):
            status = st.selectbox('檢查狀態', STATUSES,
                                  index=STATUSES.index(existing.get('status', STATUSES[0])))
            label_review = st.selectbox('原始標籤核對', LABEL_REVIEWS,
                                       index=LABEL_REVIEWS.index(existing.get('label_review', LABEL_REVIEWS[0])))
            issues = st.multiselect('觀察到的影像或資料問題（可複選）', ISSUES,
                                    default=[v for v in existing.get('image_issues', '').split('|') if v in ISSUES])
            source_note = st.text_input('資料來源／拍攝條件（不知道就寫未知）', existing.get('source_note', ''))
            patient_note = st.text_input('疑似同病人／近似照片的研究代碼或檔名', existing.get('patient_group_note', ''))
            reviewer = st.text_input('檢查者代碼（選填）', existing.get('reviewer', ''))
            notes = st.text_area('具體觀察與待確認事項', existing.get('notes', ''), height=140)
            save = st.form_submit_button('儲存這張的檢查紀錄', type='primary')
        if save:
            record = {'predictions_sha256': prediction_hash, 'image_path': row.image_path,
                      'original_label': int(row.label), 'prediction': int(row.prediction),
                      'score': float(row.score), 'error_group': row.error_group,
                      'status': status, 'label_review': label_review, 'image_issues': '|'.join(issues),
                      'source_note': source_note, 'patient_group_note': patient_note,
                      'reviewer': reviewer, 'notes': notes,
                      'updated_at_utc': datetime.now(timezone.utc).isoformat()}
            try:
                save_review(args.review_out, record, args.predictions)
            except Exception as exc:
                st.error(f'存檔失敗：{exc}')
            else:
                st.session_state['saved_notice'] = f'已儲存：{row.image_path}'
                st.rerun()
        if notice := st.session_state.pop('saved_notice', None):
            st.success(notice)
        st.caption(f'查核紀錄：{args.review_out}。按儲存後會保留在本機，可關閉頁面後繼續。')


if __name__ == '__main__':
    main()
