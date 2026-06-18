# Lensless Computational Imaging - реконструкция безлинзовых изображений

Реализация [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102) для домашнего задания №5 по курсу Глубинного обучения.

Проект построен на основе [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) с использованием [Hydra](https://hydra.cc/docs/intro/) для конфигурации и [Comet ML](https://www.comet.com/) для логирования.

## Содержание

- [Архитектура](#архитектура)
- [Структура репозитория](#структура-репозитория)
- [Установка](#установка)
- [Тренировка](#тренировка)
- [Инференс](#инференс)
- [Подсчёт метрик](#подсчёт-метрик)
- [Демо](#демо)
- [Результаты](#результаты)
- [Comet эксперименты](#comet-эксперименты)
- [Ссылки](#ссылки)

## Архитектура(TODO)

| Компонент | Параметры |
|---|---|
| ADMM (camera inversion) | расщепление переменных на $u$, $v$, $w$, $x$; апдейты Le-ADMM, паддинг $\times 2$ для работы со свёрткой |
| ADMM-100 | классический ADMM, $\mu = 10^{-4}$, $\tau = 2 \cdot 10^{-4}$, 100 итераций, без обучения (бейзлайн) |
| ADMM-Unrolled | 20 итераций, обучаемые $\{\mu_1, \mu_2, \mu_3, \tau\}$ на каждой итерации, лог-параметризация для неотрицательности параметров |
| DRUNet | Взят Apppendix.B Bezzam: strided-conv down, transposed-conv up, аддитивные skip, residual-блоки с ReLU, удвоение каналов |
| ModularReconstruction | pre-DRUNet $\rightarrow$ LeADMM (5 итераций) $\rightarrow$ post-DRUNet (`pre`/`post` опциональны, есть эксперименты с рассмотрением их по отдельности) |
| Loss | `LenslessLoss` $=MSE + LPIPS$ (VGG, `normalize = True`), на ROI кропах |
| Метрики | PSNR, SSIM, LPIPS, MSE на ROI кропах; выход модели min-max нормируется для выравнивания яркости |
| Оптимизатор | Adam, lr $= 10^{-4}$ (const), $\beta = (0.9, 0.999)$ - паарметры из статьи Bezzam, lr_sheduler = `ConstantLR` |

Три пространства изображений: **padded** $760 \times 1014$ (ADMM/FFT), **sensor** $380 \times 507$ (измерение и выход), **ROI** $200 \times 266$(лосс и метрики). PSF симулируется с помощью SLM-маски (библиотека `waveprop`).

Параметры тренировки: batch size = 2 (до 4 на A100), по 14 эпох на каждую модель, NVIDIA A100 (Colab) / NVIDIA T4 (Kaggle).

## Структура репозитория

```
.
├── src/
│   ├── configs/                      # Hydra конфиги
│   │   ├── admm100.yaml              # ADMM-100 (eval-only бейзлайн)
│   │   ├── admm_unrolled.yaml        # Le-ADMM, обучаемый
│   │   ├── modular.yaml              # модульные модели (pre / post / pre-post)
│   │   ├── inference.yaml            # конфиг инференса
│   │   ├── model/                    # admm100, admm_unrolled, modular_{pre,post,pre_post}
│   │   ├── datasets/                 # датасеты mirflickr, custom_dir
│   │   ├── dataloader/, transforms/  # dataloader и преобразования
│   │   ├── metrics/                  # PSNR, SSIM, LPIPS, MSE
│   │   └── writer/                   # Comet ML
│   ├── model/
│   │   ├── admm.py, admm_utils.py    # ADMM / Le-ADMM и операторы (H, Psi, soft-threshold)
│   │   ├── drunet.py, blocks.py      # DRUNet и структурные блоки
│   │   └── modular_recon.py          # ModularReconstruction (pre + admm + post)
│   ├── loss/lensless_loss.py         # MSE + LPIPS
│   ├── metrics/                      # psnr, ssim, lpips, mse, tracker
│   ├── datasets/
│   │   ├── mirflickr_dataset.py      # DigiCam-Mirflickr (HuggingFace)
│   │   ├── custom_dir_dataset.py     # CustomDirDataset для произвольной папки
│   │   └── collate.py                # collate_fn, inference_collate_fn
│   ├── trainer/                      # trainer.py, base_trainer.py, inferencer.py
│   └── logger/                       # Comet ML writer
├── lensless_helpers/                 # препроцессинг, PSF, утилиты (crop_roi, resize, ...)
├── train.py                          # точка входа для тренировки
├── inference.py                      # инференс: сохранение реконструкций по id
├── calculate_metrics.py              # подсчёт метрик (две режима, см. ниже)
├── model_ids.py                      # Google Drive id чекпоинтов
├── Demo.ipynb                        # демо-ноутбук (Colab)
├── demo_dataset/                     # мини демо-датасет из 3 примеров (lensless/masks/lensed)
├── metrics_notebooks/                # LenslessMetrics.ipynb (финальные метрики)
├── train_notebooks/                  # ноутбуки тренировки (Colab / Kaggle)
├── sanity_check/                     # локальные sanity тесты ADMM и датасета
├── requirements.txt
├── README.md                         # этот файл
└── Report.md                         # отчёт по работе и сложностям
```

## Установка

```bash
# Клонирование репозитория
git clone -b lensless-base https://github.com/DommeUse/Lensless-Computational-Imaging.git
%cd Lensless-Computational-Imaging

# Установка виртуального окружения для корректной установки зависимостей
python -m pip install virtualenv
python -m virtualenv /content/lensless_env

# Установка зависимостей и библиотеки gdown для работы с Google Drive
/content/lensless_env/bin/pip install -r requirements.txt
/content/lensless_env/bin/pip install -q gdown
```

## Тренировка


### Данные

Датасет [bezzam/DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K) скачивается автоматически с HuggingFace при первом запуске - отдельной подготовки данных не требуется.

### Comet ML

API-ключ лучше положить в переменную окружения:
```bash
export COMET_API_KEY=<your_key>
```
В Kaggle/Colab ключ удобно хранить в secrets (примеры - в ноутбуках `train_notebooks/`).

### Sanity check

Проверка, что структурные компоненты работают корректно.
В `sanity_check/` есть локальные тесты `test_admm.py` и `test_dataset.py`:
```bash
# После установки
PYTHONPATH=. /content/lensless_env/bin/python sanity_check/test_dataset.py
PYTHONPATH=. /content/lensless_env/bin/python sanity_check/test_admm.py
```

Демонстрация работы есть в ноутбуке `sanity_check/sanity_checks.ipynb`.

### Полная тренировка

**Модульные модели** (config `modular`, по умолчанию `model=modular_pre_post`):
```bash
/content/lensless_env/bin/python train.py \
  --config-name=modular \
  model=modular_pre_post \                 # для обучения односторонней модели modular_pre и modular_post
  trainer.save_dir="<save-dir-name>" \      # директория для выхлопа обучения
  writer.log_checkpoints=True \            # флаг для сохранения чекпоинтов
  writer.run_name="<exp-name>" # имя эксперимента
```

**Unrolled Le-ADMM**:
```bash
/content/lensless_env/bin/python train.py \
  --config-name=admm_unrolled \
  trainer.save_dir="<save-dir-name>" \  # директория для выхлопа обучения
  writer.log_checkpoints=True \         # флаг для сохранения чекпоинтов
  writer.run_name="<exp-name>"          # имя эксперимента
```

### Полезные оверрайды: 
`dataloader.batch_size=4`, `optimizer.lr=3e-4`, `trainer.max_grad_norm=1.0` (gradient clipping заметно ускоряет сходимость), `trainer.resume_from=<dir>` для продолжения обучения между сессиями.

> ADMM-100 - это eval-only бейзлайн (обучаемых параметров нет), его метрики считаются скриптом `calculate_metrics.py` с `--config-name=admm100`.

### Kaggle / Colab

Готовые ноутбуки тренировки - в `train_notebooks/LenslessTrainColab.ipynb` и `train_notebooks/LenslessTrainKaggle.ipynb`.

## Инференс

### Загрузка модели

Предобученные чекпоинты моделей загружен на Google Drive, поэтому сначала нам нужно их скачать с помощью библиотеки gdown

```bash
pip install -q gdown
```

Далее выполняем код

```python
import gdown, os
from model_ids import MODEL_IDS

label = "pre-post" # выбираем модель, которую хотим протестировать
MODEL_GDRIVE_ID, MODEL_CONFIG = MODEL_IDS[label]

CHECKPOINT_DIR = "checkpoints" # выбераем название директории, куда загружать чекпоинты
CHECKPOINT_NAME = "model_best" # выбераем имя загружаемого чекпоинта

os.makedirs(CHECKPOINT_DIR, exist_ok = True)
CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/{CHECKPOINT_NAME}.pth'

gdown.download(id = MODEL_GDRIVE_ID, output = CHECKPOINT_PATH, quiet = False)
```

### Запуск модели

`inference.py` применяет модель к датасету и сохраняет по одной реконструкции `<id>.png` на каждый вход (id совпадает с id входного изображения).

**Кастомная директория** (`CustomDirDataset`). Путь задаётся через `datasets.data_dir`:
```bash
/content/lensless_env/bin/python inference.py --config-name=inference \
    model={MODEL_CONFIG} \
    from_pretrained={CHECKPOINT_PATH} \
    datasets.data_dir="<path_to_NameOfTheDirectoryWithData>" \
    +save_dir=reconstructions # имя директории для сохранения результатов
```

Ожидаемый формат кастомной директории:
```
NameOfTheDirectoryWithData/
├── lensless/   ImageID1.png ...   # обязательно: безлинзовые изображения
├── masks/      ImageID1.npy ...   # обязательно: значения SLM-маски в формате .npy
└── lensed/     ImageID1.png ...   # опционально: ground-truth оригиналы
```

**Встроенный датасет** (например, тестовый сплит Mirflickr):

Также inference.py поддерживает тестирование подели на тестовой выборке датасета Mirflickr.

> Только будьте осторожны, инференс длится на порядок дольше из-за большого размера датасета
```bash
python inference.py --config-name=inference \
    model={MODEL_CONFIG} \
    from_pretrained={CHECKPOINT_PATH} \
    datasets=mirflickr partition=test \
    +save_dir=reconstructions # имя директории для сохранения результатов
```

Реконструкции сохраняются в `save_dir` (по умолчанию `reconstructions/`).

## Подсчёт метрик

`calculate_metrics.py` работает в **двух режимах**.

**Режим 1 - оценка модели на сплите датасета** (PSNR / SSIM / LPIPS / MSE на test-сплите + число параметров):
```bash
/content/lensless_env/bin/python calculate_metrics.py \
    --config-name=modular \
    model=modular_pre_post \ # для оценки односторонней модели modular_pre и modular_post
    ~writer \ # 
    +from_pretrained=checkpoints/model_best.pth
```
Для ADMM-100: `--config-name=admm100`. Для unrolled: `--config-name=admm_unrolled`.

Для логирования метрик и тестовых изображений в CometML передайте также аргумент `writer.run_name="<run-name>"` вместо `~writer`. Не забудьте перед этим установить `COMET_API_KEY` в ваше окружение.

**Режим 2 - сравнение сохранённых реконструкций с ground truth** (включается, когда заданы `gt_dir` и `recon_dir`; модель не грузится, изображения матчатся по id):
```bash
/content/lensless_env/bin/python calculate_metrics.py \
    ~writer \
    +gt_dir="<path_to_NameOfTheDirectoryWithData>" \
    +recon_dir=reconstructions # имя директории с восстановленными изображениями
```
`gt_dir` должна указывать на путь в папку `lensed/`, то есть прямо на папку с оригиналами. `~writer` отключает Comet. Если ground truth нет - скрипт сообщает об этом и завершается без ошибки.

## Демо

[`Demo.ipynb`](./Demo.ipynb) - end-to-end демонстрация в Google Colab: клонирует репозиторий, ставит зависимости, скачивает чекпоинт, принимает ссылку на `.zip`-датасет в Google Drive, запускает инференс, визуализирует примеры (оригинал, если есть, vs безлинзовое vs реконструкция) и печатает метрики. Для быстрой проверки в репозитории лежит `demo_dataset/` из 3 примеров.

## Результаты

> Финальные метрики считаются в [`metrics_notebooks/LenslessMetrics.ipynb`](./metrics_notebooks/LenslessMetrics.ipynb).

| Модель | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | MSE $\downarrow$ |
|---|---|---|---|---|
| ADMM-100(бейзлайн) | **10.898** | **0.279** | **0.752** | **0.086** |
| ADMM-Unrolled | **10.997** | **0.176** | **0.753** | **0.089** |
| Modular Pre | **13.335** | **0.219** | **0.634** | **0.050** |
| Modular Post | **15.953** | **0.431** | **0.550** | **0.028** |
| Modular Pre-Post | **16.554** | **0.458** | **0.527** | **0.025** |

Подробное обсуждение результатов и сложностей - в [Report.md](./Report.md).

## Comet эксперименты

| Модель | Train | Inference |
|---|---|---|
| ADMM-100 | - | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/0eae0d03acae41f79c727201d249bb48) |
| ADMM-Unrolled | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/yor68kxc3rq083std9sp4ievxefzd6jn) | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/8624e4d5dda34ee1adf96c642c3cec2b) |
| Modular Pre-Post | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/sobsv72s4yhtocgfiagbym6u40z6qfef) | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/a1341e4daa1d49af836238e847a4e369) |
| Modular Only-Pre | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/nzgn47ok8tqnwlz7lme6d6nasltaitwz) | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/281b910679f0435daed50a803549d054) |
| Modular Only-Post | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/a5z5pkiyealbs5c58giiw2slih9v4vho) | [link](https://www.comet.com/german-zverev/lensless-computational-imaging/d40ad79a9312448a9eb8fafce9afef31) |

## Ссылки

- Шаблон проекта: [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) by [Petr Grinberg](https://github.com/Blinorot)
- Статья Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction: E. Bezzam, Y. Perron, M. Vetterli, 2025, [arXiv:2502.01102](https://arxiv.org/abs/2502.01102)
- Статья Learned reconstructions for practical mask-based lensless imaging: K. Monakhova, J. Yurtsever, G. Kuo, N. Antipa, K. Yanny, L. Waller, 2019, [arXiv:2009.02095](https://arxiv.org/abs/2009.02095)