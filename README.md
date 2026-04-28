## Сегментация клеток дрожжей

### Получение датасета

Скачиваем датасет и распаковываем в папку data:
```
cd data
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3799/yeast_cell_in_microstructures_dataset.zip
unzip yeast_cell_in_microstructures_dataset.zip -d yeast_cell_in_microstructures_dataset
cd ../
```

### Создание окружения
Перед дальнейшими действиями деактивируйте все активные окружения conda/venv.

Создайте виртуальное окружение и установите зависимости (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Выберите файл с зависимостями под вашу платформу:

```bash
# macOS
pip install -r requirements-macos.txt

# Linux (CPU only)
pip install -r requirements-linux-cpu.txt

# Linux + CUDA
pip install -r requirements-linux-cuda.txt
```

Дополнительно для разработки (форматтеры, линтеры):
```bash
pip install -r requirements-dev.txt
```

### Проверка: запуск обучения
```
python -m scripts.train
```

В другом окне терминала из корня проекта запустите Tensorboard — инструмент для визуализации логов обучения.
```
tensorboard --logdir logs/
```
В браузере по адресу http://localhost:6006/ откроется панель Tensorboard, где можно увидеть значения метрик, примеры предсказаний и прочие артефакты обучения моделей.

Чекпоинты модели будут сохраняться в `logs/runs/DATE/TIME/checkpoints/`

### Запуск в Google Colab
Если у вас нет Nvidia GPU или Macbook M1+, обучение моделей может занять много времени. В этом случае написание кода рекомендуется делать локально и пушить в репозиторий, а запуск обучения можно сделать в Colab. Инструкция, как это можно сделать: [Запуск в Google Colab](colab.md)

### Задания

Результатом выполнения заданий является репозиторий с кодом и небольшой отчёт в README.md

Что добавить в отчёт:
- Краткое описание того, что реализовано, с мотивацией для каждого эксперимента.
- График обучения (`val/loss`, `val/Dice`) — сравнение с baseline.
- Примеры предсказаний.
- Сравнительный анализ экспериментов: динамика обучения, метрики, визуальное качество.

#### Задание 1. Soft Dice Loss
1. Расширьте функцию `build_loss` в `cell_segmentation/factory.py` так, чтобы поддерживалась альтернативная функция ошибки — Soft Dice Loss, реализованная в `cell_segmentation/utils/losses.py`. Выбор функции должен задаваться полем `loss.name` в конфиге `config/train.yaml`, по аналогии с тем, как через `model.name` выбирается модель.
2. Запустите эксперименты с Cross Entropy и Soft Dice Loss, сравните значения метрик и качественно оцените различия в предсказаниях.
3. Сохраните чекпоинты лучших моделей в папку `weights` и запустите `scripts/inference.py` для каждой из них. Перед этим обновите поле `checkpoint` в `config/inference.yaml`. Если скрипт отрабатывает без ошибки и `test/Dice > 0.95` — всё получилось.

#### Задание 2. Изменение архитектуры модели
Ваша задача — извлечь максимальную точность из модели, в которой будет не более 50k параметров. Для этого внесите правки в архитектуру модели.

В частности, попробуйте заменить в модуле Down комбинацию conv+maxpool на dilated conv (`nn.Conv2d(..., dilation=2)`) с одновременным увеличением шага кернела (параметр `stride`). В остальном — полная свобода творчества. Реализацию делайте в `cell_segmentation/model/tiny_unet.py`, имя `tiny_unet` уже зарегистрировано в словаре `MODELS` в `cell_segmentation/factory.py`.

Для экспериментов используйте конфиг `config/train_tiny.yaml`. Запуск:
```
python -m scripts.train --config-name train_tiny
```

#### Задание 3 (опционально). Идеи из статей
Этот блок — про чтение статьи и реализацию идеи из неё. Цель не «обогнать baseline» (на этом датасете все три варианта дают `test/Dice ≈ 0.97` — в пределах шума на 49 валидационных примерах), а **разобраться в подходе**, описать наблюдения (скорость сходимости, визуальные артефакты в Tensorboard, динамика метрик) и ответить, в каких ситуациях этот приём вообще должен помогать. LLM-ассистенты разрешены.

Базовая модель для сравнений: UNet + Cross Entropy, 30 эпох

Вы можете выбрать несколько вариантов. Реализацию пишете сами, разрешено пользоваться LLM. После реализации не забудьте зарегистрировать имя в соответствующем словаре в `cell_segmentation/factory.py` (`MODELS` для моделей, ветка `if name == ...` в `build_loss` для функций ошибки).

1. **Focal loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)). Идея: понижать вес «лёгких» (хорошо предсказанных) пикселей, чтобы модель училась на сложных. Реализуйте класс `FocalLoss(nn.Module)` в `cell_segmentation/utils/losses.py` и зарегистрируйте имя `focal` в `build_loss`. Параметр `gamma` (обычно 1–5) задайте через `loss.gamma` в конфиге. Запуск: `python -m scripts.train loss.name=focal loss.gamma=2.0`.
2. **Squeeze-and-Excitation U-Net** ([Hu et al. 2018](https://arxiv.org/abs/1709.01507)). Идея: для каждого канала свёрточной активации обучить скалярный вес («внимание по каналам») для их адаптивной перекалибровки. Создайте файл `cell_segmentation/model/unet_se.py` с классом `UNet`, добавьте `unet_se` в `MODELS`. Запуск: `python -m scripts.train model.name=unet_se`.
3. **Attention U-Net** ([Oktay et al. 2018](https://arxiv.org/abs/1804.03999)). Идея: применить пространственный attention gate, который маскирует «нерелевантные» области входных данных. Создайте файл `cell_segmentation/model/unet_attention.py` с классом `UNet`, добавьте `unet_attention` в `MODELS`. Запуск: `python -m scripts.train model.name=unet_attention`.

#### Задание 4 (опционально). Детекция клеток
В пакете `cell_detection/` подготовлена базовая реализация YOLO-подобной модели для детекции клеток на тех же данных. Архитектура: компактный backbone (`TinyYOLO`), 1 якорь по умолчанию, выход на сетке 8×8 ячеек. Метрики: `mAP` (по умолчанию IoU=0.50:0.95) через `torchmetrics`. Базовая идея модели — статья [YOLOv1 (Redmon et al. 2016)](https://arxiv.org/abs/1506.02640).

Запуск:
```
python -m scripts.train_detection
```

Ваши задачи (по мотивам [учебной тетрадки по детекции](https://github.com/norsage/dl-mcs/blob/main/notebooks/object_detection.ipynb) и [тетрадки с заданиями](https://github.com/norsage/dl-mcs/blob/main/assignments/cell_detection.ipynb)):

1. **Базовое обучение.** Запустите обучение на 30–50 эпох с дефолтными гиперпараметрами и убедитесь, что `val/map > 0.3`.
2. **Кластеризация якорей** ([YOLO9000, Redmon & Farhadi 2016](https://arxiv.org/abs/1612.08242), раздел *Dimension clusters*). По обучающей выборке найдите подходящие размеры якорей через KMeans (k=3–5). Перепишите поле `data.anchors` в конфиге и переобучите. KMeans можно реализовать самостоятельно в ~10 строк numpy либо поставить `scikit-learn` отдельно (`pip install scikit-learn`).
3. **Максимальная точность.** Цель — `val/map > 0.6` любыми средствами. Темы ниже на курсе подробно не разбирались, поэтому здесь **разрешается активно пользоваться LLM-ассистентами** (Claude / ChatGPT / Cursor) — как для понимания идеи, так и для генерации кода. Идеи, которые обычно дают видимый эффект:
    - **Больше якорей.** Замените единственный `[48, 72]` на 3–5 якорей разных пропорций (естественно сочетается с заданием 2). Не забудьте обновить `len(anchors)` в инициализации модели.
    - **Аугментации.** Добавьте случайные флипы / сдвиги / контраст в `cell_detection/data/dataset.py`. Внимание: при флипе изображения координаты рамок тоже должны переворачиваться. Для удобства можно поставить [`albumentations`](https://albumentations.ai/) — она умеет согласованно преобразовывать изображения и bbox-разметку.
    - **LR scheduler.** Cosine annealing (`torch.optim.lr_scheduler.CosineAnnealingLR`) или OneCycle (`OneCycleLR`) — добавьте в `configure_optimizers` в `DetectionModel`.
    - **Focal loss для objectness** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)). Замените `BCEWithLogitsLoss` для no-object loss на focal loss — это помогает в ситуациях, когда фоновых ячеек гораздо больше, чем объектных.
    - **GIoU / CIoU loss** ([Rezatofighi et al. 2019](https://arxiv.org/abs/1902.09630), [Zheng et al. 2020](https://arxiv.org/abs/1911.08287)) вместо MSE для координат рамки. В `torchvision.ops` есть готовые `generalized_box_iou_loss` и `complete_box_iou_loss`.
    - **Гиперпараметры.** Перебор `lr`, `weight_decay`, размер батча; больше эпох (100+).
