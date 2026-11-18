"""Run: streamlit run app.py -- --checkpoint runs/b0_v1/model.pt"""
import argparse
from pathlib import Path
import streamlit as st
import torch
from dfu import load_checkpoint, predict_image, read_rgb

st.set_page_config(page_title='糖尿病足影像研究', page_icon='🔬', layout='centered')
st.title('糖尿病足影像轉診研究')
st.caption('EfficientNet-B0 · 本機單張影像預測')
st.info('研究展示用，結果需由醫師覆核；模型分數不是已校正的臨床風險百分比。')
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='runs/b0_v1/model.pt')
args, _ = p.parse_known_args()
path = Path(args.checkpoint)

if not path.is_file():
    st.warning('尚未找到已訓練的模型。請先依操作說明完成訓練，再啟動此頁面。')
    st.stop()


@st.cache_resource
def cached_model(filename, modified_ns):
    torch.set_num_threads(2)
    return load_checkpoint(filename, 'cpu')


try:
    model, checkpoint = cached_model(str(path.resolve()), path.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'模型讀取失敗：{exc}')
    st.stop()
if checkpoint['initialization'] == 'random_test_only':
    st.warning('這是用於測試程式的隨機初始化模型，不能用於展示研究成效。')
st.write('標籤定義：', checkpoint['label_definition'])
if not checkpoint.get('patient_separation_verified', True):
    st.info('此模型使用缺少病人 ID 的既有影像分組；尚未確認不同組的病人互不重疊。')
uploaded = st.file_uploader('選擇一張傷口照片', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    try:
        image = read_rgb(uploaded)
        st.image(image, caption='待預測影像', use_container_width=True)
        if st.button('執行模型預測', type='primary'):
            uploaded.seek(0)
            result = predict_image(model, checkpoint, uploaded, torch.device('cpu'))
            text = '需轉診' if result['prediction'] == 1 else '不需轉診'
            st.subheader(f'模型預測：{text}')
            a, b = st.columns(2)
            a.metric('轉診分數（0–1）', f"{result['score']:.3f}")
            b.metric('驗證集選定門檻', f"{result['threshold']:.3f}")
            st.caption('單張照片預測不等於完整臨床評估。照片不會由本程式寫入硬碟。')
    except Exception as exc:
        st.error(f'無法處理這張照片：{exc}')
