# Использована LLM: Claude Opus 5.5
"""Демо ДЗ-3: отбор «хороших» вин (оценка экспертов ≥ 7) логистической регрессией.

Запуск: streamlit run app.py
Модель здесь не обучается: приложение только загружает пайплайн, порог и тестовую выборку,
сохранённые ноутбуком HW3_MMDA.ipynb (model.joblib, config.json, test.csv).
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

# Пути строим от папки с app.py, а не от текущей директории: приложение запускается откуда угодно
# (`streamlit run hw_3/wine_app/app.py`, Streamlit Community Cloud) и всё равно находит свои файлы.
APP_DIR = Path(__file__).parent
TARGET = 'target'
FEATURE_INFO = {  # столбец: (подпись, шаг ввода, формат)
    'fixed_acidity': ('Фиксированная кислотность, г/л (винная кислота)', 0.1, '%.1f'),
    'volatile_acidity': ('Летучая кислотность, г/л (уксусная кислота)', 0.01, '%.2f'),
    'citric_acid': ('Лимонная кислота, г/л', 0.01, '%.2f'),
    'residual_sugar': ('Остаточный сахар, г/л', 0.1, '%.1f'),
    'chlorides': ('Хлориды, г/л (хлорид натрия)', 0.001, '%.3f'),
    'free_sulfur_dioxide': ('Свободный диоксид серы, мг/л', 1.0, '%.0f'),
    'total_sulfur_dioxide': ('Общий диоксид серы, мг/л', 1.0, '%.0f'),
    'density': ('Плотность, г/см³', 0.0001, '%.4f'),
    'pH': ('pH', 0.01, '%.2f'),
    'sulphates': ('Сульфаты, г/л (сульфат калия)', 0.01, '%.2f'),
    'alcohol': ('Алкоголь, % об.', 0.1, '%.1f'),
    'is_red': ('Цвет вина', None, None),
}
COLOR_FOR = '#2a78d6'      # «за» хорошее вино; основная серия
COLOR_AGAINST = '#eb6834'  # «против»; вторая серия
COLOR_MUTED = '#52514e'

st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')


# ---------------------------------------------------------------- 4.5. Кэширование
# Почему так (подробнее — README.md):
# * st.cache_resource хранит ОДИН объект на весь сервер и отдаёт всем сессиям ссылку на него без
#   копирования и без сериализации. Это подходит для модели: она тяжёлая, только читается, и копия
#   на каждый вызов не нужна. Без декоратора joblib.load читал бы файл с диска и заново собирал
#   пайплайн при КАЖДОМ перезапуске скрипта, т. е. при каждом движении слайдера.
# * st.cache_data хранит результат как данные: возвращает каждой сессии свою копию (через pickle),
#   и одна сессия не испортит кэш другой, если изменит таблицу. Это подходит для DataFrame и массивов.
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    """Вероятности класса 1 на test. От порога не зависят, поэтому считаются один раз.

    Аргумент _model начинается с подчёркивания: Streamlit не хеширует его, ключ кэша — только X.
    """
    return _model.predict_proba(X)[:, 1]


def threshold_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else float('nan')
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fp / (fp + tn)}


def prepare_batch(frame, features):
    """Проверяет загруженную таблицу. Возвращает (X, ошибка, проигнорированные столбцы)."""
    missing = [column for column in features if column not in frame.columns]
    if missing:
        return None, ('В файле не хватает столбцов: ' + ', '.join(f'`{column}`' for column in missing)
                      + '. Нужны все 12 признаков: ' + ', '.join(f'`{column}`' for column in features) + '.'), []
    if frame.empty:
        return None, 'В файле нет ни одной строки с данными.', []
    X = frame[features].apply(pd.to_numeric, errors='coerce')
    bad = X.isna()
    if bad.any().any():
        rows = (bad.any(axis=1).to_numpy().nonzero()[0] + 1)[:10].tolist()
        columns = [column for column in features if bad[column].any()]
        return None, (f'Пустые или нечисловые значения в столбцах {", ".join(f"`{c}`" for c in columns)} '
                      f'(строки данных {rows}{" и др." if bad.any(axis=1).sum() > 10 else ""}). '
                      'Исправьте файл и загрузите снова.'), []
    ignored = [column for column in frame.columns if column not in features]
    return X, None, ignored


model = load_model()
test = load_test()
config = load_config()
features = list(model.feature_names_in_)
X_test, y_test = test[features], test[TARGET].to_numpy()
default_threshold = float(config['threshold'])

# ---------------------------------------------------------------- сайдбар: общий порог
# Ключ виджета 'threshold' хранит значение в st.session_state, поэтому порог один для всех вкладок.
if 'threshold' not in st.session_state:
    st.session_state.threshold = default_threshold


def reset_threshold():
    st.session_state.threshold = default_threshold


with st.sidebar:
    st.header('🍷 Порог решения')
    st.slider('Порог вероятности «хорошего» вина', min_value=0.0, max_value=1.0, step=0.001, format='%.3f',
              key='threshold',
              help='Вино попадает в подборку, если вероятность того, что оно хорошее, не ниже порога. '
                   'Порог общий для всех вкладок.')
    st.button(f'Вернуть порог из config.json ({default_threshold:.3f})', on_click=reset_threshold)
    st.caption(f'По умолчанию — порог, выбранный в ноутбуке на валидации: {default_threshold:.4f} '
               '(максимальная precision при recall ≥ 0.7).')
threshold = float(st.session_state.threshold)

st.title('Подборка вин: какие вина считать хорошими')
st.caption('Логистическая регрессия по 11 физико-химическим характеристикам и цвету вина. '
           '«Хорошее» вино — оценка экспертов не ниже 7 из 10 (датасет Wine Quality, UCI).')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

# ---------------------------------------------------------------- 4.1. Оценка одного вина
with tab_one:
    st.subheader('Оценка одного вина')
    st.write('Диапазоны полей — минимум и максимум по тестовой выборке, значения по умолчанию — медианы.')
    stats = X_test.agg(['min', 'max', 'median'])
    values = {}
    color = st.radio(FEATURE_INFO['is_red'][0], ['белое', 'красное'], index=int(stats.loc['median', 'is_red']),
                     horizontal=True, key='input_is_red')
    values['is_red'] = int(color == 'красное')
    numeric = [column for column in features if column != 'is_red']
    columns = st.columns(3)
    for i, column in enumerate(numeric):
        label, step, fmt = FEATURE_INFO.get(column, (column, None, None))
        with columns[i % 3]:
            values[column] = st.number_input(label, min_value=float(stats.loc['min', column]),
                                             max_value=float(stats.loc['max', column]),
                                             value=float(stats.loc['median', column]),
                                             step=step, format=fmt, key=f'input_{column}')
    wine = pd.DataFrame([values])[features]
    proba_one = float(model.predict_proba(wine)[0, 1])
    left, right = st.columns([1, 2])
    left.metric('Вероятность, что вино хорошее', f'{proba_one:.1%}')
    with right:
        if proba_one >= threshold:
            st.success(f'**Рекомендовать в подборку**: вероятность {proba_one:.3f} ≥ порога {threshold:.3f}.')
        else:
            st.warning(f'**Не включать в подборку**: вероятность {proba_one:.3f} < порога {threshold:.3f}.')

# ---------------------------------------------------------------- 4.2. Модельные метрики
with tab_metrics:
    st.subheader('Модельные метрики на тестовой выборке')
    proba = test_proba(model, X_test)
    current = threshold_metrics(y_test, proba, threshold)
    base = threshold_metrics(y_test, proba, default_threshold)
    selected = current['tp'] + current['fp']
    st.write(f'Тестовая выборка: {len(y_test)} вин, из них хороших — {int(y_test.sum())}. '
             f'Текущий порог: **{threshold:.3f}**.')

    cols = st.columns(4)
    for col, key, label in zip(cols, ['precision', 'recall', 'f1'], ['Precision', 'Recall', 'F1']):
        delta = current[key] - base[key]
        col.metric(label, f'{current[key]:.3f}' if np.isfinite(current[key]) else '—',
                   delta=f'{delta:+.3f} к порогу из config' if np.isfinite(delta) and threshold != default_threshold
                   else None)
    cols[3].metric('Вин в подборке', f'{selected} из {len(y_test)}')

    confusion = pd.DataFrame([[current['tn'], current['fp']], [current['fn'], current['tp']]],
                             index=['истина: не хорошее (0)', 'истина: хорошее (1)'],
                             columns=['модель: не брать (0)', 'модель: в подборку (1)'])
    left, right = st.columns([1, 2])
    with left:
        st.markdown('**Матрица ошибок**')
        st.table(confusion)
        st.caption(f'FP = {current["fp"]}: посредственные вина в подборке. '
                   f'FN = {current["fn"]}: хорошие вина, которые не попали в подборку.')
    with right:
        fpr_curve, tpr_curve, _ = metrics.roc_curve(y_test, proba)
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_test, proba)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        axes[0].plot(fpr_curve, tpr_curve, color=COLOR_FOR, linewidth=2,
                     label=f'модель, ROC-AUC = {metrics.roc_auc_score(y_test, proba):.3f}')
        axes[0].plot([0, 1], [0, 1], color=COLOR_MUTED, linestyle=':', linewidth=1.5, label='случайная модель')
        axes[0].scatter(current['fpr'], current['recall'], s=90, color=COLOR_AGAINST, edgecolor='white',
                        linewidth=2, zorder=3, label=f'порог {threshold:.3f}')
        axes[0].set_xlabel('FPR: какая доля посредственных вин попала в подборку')
        axes[0].set_ylabel('TPR = recall: какая доля хороших вин попала в подборку')
        axes[0].set_title('ROC-кривая')
        axes[1].plot(recall_curve, precision_curve, color=COLOR_FOR, linewidth=2, drawstyle='steps-post',
                     label=f'модель, PR-AUC = {metrics.average_precision_score(y_test, proba):.3f}')
        axes[1].axhline(y_test.mean(), color=COLOR_MUTED, linestyle=':', linewidth=1.5,
                        label=f'случайная модель ({y_test.mean():.3f})')
        if np.isfinite(current['precision']):
            axes[1].scatter(current['recall'], current['precision'], s=90, color=COLOR_AGAINST, edgecolor='white',
                            linewidth=2, zorder=3, label=f'порог {threshold:.3f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title('PR-кривая')
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right' if ax is axes[0] else 'upper right', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    if not np.isfinite(current['precision']):
        st.info('При таком пороге в подборку не попадает ни одно вино, поэтому precision не определена.')

    st.markdown(
        f"""**Сомелье: как порог меняет подборку**

Модель выдаёт каждому вину вероятность того, что оно хорошее, и в подборку попадают вина с вероятностью
не ниже порога.

- **Порог выше** — подборка короче и «строже». Среди отобранных вин больше действительно хороших (precision
  растёт), и клиенты реже разочаровываются. Но больше хороших вин остаётся за бортом (recall падает),
  и подборка становится беднее.
- **Порог ниже** — подборка шире: в неё попадает больше хороших вин (recall растёт), но вместе с ними и больше
  посредственных (precision падает).

Сейчас при пороге {threshold:.3f} в подборку попадают {selected} вин из {len(y_test)}: хороших среди них
{current['tp']}, посредственных — {current['fp']}. В подборку попали {current['recall']:.0%} всех хороших вин."""
    )

# ---------------------------------------------------------------- 4.3. Пакетное предсказание
with tab_batch:
    st.subheader('Пакетное предсказание')
    st.write(f'Загрузите CSV с 12 признаками вин. Порог берётся из сайдбара: **{threshold:.3f}**. '
             'Столбец `target` и другие лишние столбцы не используются.')
    example = X_test.head(20).to_csv(index=False).encode('utf-8')
    st.download_button('Скачать пример файла (20 вин из test.csv)', data=example, file_name='wines_example.csv',
                       mime='text/csv')
    uploaded = st.file_uploader('CSV-файл с признаками вин', type='csv')
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            st.error(f'Не удалось прочитать CSV: {error}')
        else:
            X_batch, error, ignored = prepare_batch(batch, features)
            if error:
                st.error(error)
            else:
                if ignored:
                    st.info('Не используются при предсказании: ' + ', '.join(f'`{column}`' for column in ignored))
                proba_batch = model.predict_proba(X_batch)[:, 1]
                result = batch.copy()
                result['вероятность_хорошее'] = proba_batch.round(4)
                result['решение'] = np.where(proba_batch >= threshold, 'в подборку', 'не брать')
                n_selected = int((proba_batch >= threshold).sum())
                st.write(f'Вин в файле: {len(result)}, рекомендовано в подборку: **{n_selected}**.')
                st.dataframe(result, width='stretch')
                buffer = io.StringIO()
                result.to_csv(buffer, index=False)
                st.download_button('Скачать таблицу с решениями (CSV)', data=buffer.getvalue().encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv')

# ---------------------------------------------------------------- 4.4. Что выучила модель
with tab_coef:
    st.subheader('Что выучила модель')
    logreg = model[-1]
    coef = pd.Series(logreg.coef_.ravel(), index=features).sort_values()
    labels = [FEATURE_INFO.get(name, (name,))[0].split(',')[0] + f'  ({name})' for name in coef.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_FOR if value > 0 else COLOR_AGAINST for value in coef]
    ax.barh(labels, coef.to_numpy(), color=colors, height=0.6)
    ax.axvline(0, color=COLOR_MUTED, linewidth=1)
    for y_pos, value in enumerate(coef):
        ax.annotate(f'{value:+.2f}', (value, y_pos), xytext=(4 if value > 0 else -4, 0), textcoords='offset points',
                    ha='left' if value > 0 else 'right', va='center', fontsize=9, color='#333333')
    span = np.abs(coef).max() * 1.25
    ax.set_xlim(-span, span)
    ax.set_xlabel('Коэффициент: изменение логарифма шансов «хорошего» вина при росте признака на 1 std')
    ax.set_title('Коэффициенты логистической регрессии после StandardScaler')
    ax.legend(handles=[Patch(color=COLOR_FOR, label='«за» хорошее вино (коэффициент > 0)'),
                       Patch(color=COLOR_AGAINST, label='«против» (коэффициент < 0)')], loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    positive, negative = coef[coef > 0][::-1], coef[coef < 0]
    st.markdown(
        '**Сильнее всего «за» хорошее вино:** '
        + ', '.join(f'`{name}` ({value:+.2f})' for name, value in positive.head(3).items())
        + '.  \n**Сильнее всего «против»:** '
        + ', '.join(f'`{name}` ({value:+.2f})' for name, value in negative.head(3).items()) + '.'
    )
    st.caption('Признаки стандартизированы, поэтому коэффициенты сравнимы: это изменение логарифма шансов при росте '
               'признака на одно стандартное отклонение (std по train) при прочих равных. Осторожно с '
               'интерпретацией: плотность почти полностью определяется сахаром и алкоголем, поэтому знаки у `density` '
               'и `residual_sugar` отражают их совместный вклад, а не эффект каждого признака по отдельности. '
               f'Свободный член: {logreg.intercept_[0]:+.2f}.')
