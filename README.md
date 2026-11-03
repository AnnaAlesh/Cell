\# Сегментация клеток дрожжей (Cell Segmentation)



Учебный проект по семантической сегментации клеток дрожжей в микроструктурах. В рамках работы реализованы альтернативные функции потерь и модификации архитектуры нейронных сетей.



\---



\## Содержание



1\. \[Установка и запуск](#установка-и-запуск)

2\. \[Задание 1: Soft Dice Loss](#задание-1-soft-dice-loss)

3\. \[Задание 2: Изменение архитектуры (tiny\_unet)](#задание-2-изменение-архитектуры-tiny\_unet)

4\. \[Задание 3: (Опционально)](#задание-3-опционально)



\---



\## Установка и запуск



1\. Клонируйте репозиторий.

2\. Установите зависимости:



&#x20;  ```bash

&#x20;  pip install -r requirements-windows.txt

&#x20;  ```



3\. Скачайте датасет и распакуйте его в папку `data/yeast\_cell\_in\_microstructures\_dataset`.



\*\*Запуск обучения:\*\*



```bash

python -m scripts.train loss.name=cross\_entropy  # или

python -m scripts.train loss.name=soft\_dice

```



\*\*Запуск инференса:\*\*



```bash

python scripts/inference.py

```



\---



\## Задание 1: Soft Dice Loss



\*\*Что сделано:\*\*



В файл `cell\_segmentation/factory.py` добавлена поддержка функции потерь `SoftDiceLoss` (из `cell\_segmentation/utils/losses.py`). Теперь выбор функции задается полем `loss.name` в `config/train.yaml`.



\*\*Мотивация:\*\*



Cross Entropy оптимизирует попиксельную точность, но не всегда хорошо справляется с дисбалансом классов. Soft Dice напрямую оптимизирует метрику пересечения (Dice), что часто дает более гладкие и связные маски.



\### Эксперимент 1: Cross Entropy (baseline)



\- \*\*val/Dice:\*\* 0.949

\- \*\*test/Dice:\*\* 0.952

\- \*\*test/loss:\*\* 0.266



!\[val\_dice\_ce](assets/val\_dice\_ce.png)



\*Рис. 1. Динамика val/Dice для Cross Entropy.\*



!\[val\_loss\_ce](assets/val\_loss\_ce.png)



\*Рис. 2. Динамика val/loss для Cross Entropy.\*



!\[predictions\_ce](assets/predictions\_ce.png)



\*Рис. 3. Примеры предсказаний модели с Cross Entropy.\*



\### Эксперимент 2: Soft Dice Loss



\- \*\*val/Dice:\*\* 0.954

\- \*\*test/Dice:\*\* 0.955

\- \*\*test/loss:\*\* 0.288



!\[val\_dice\_soft](assets/val\_dice\_soft.png)



\*Рис. 4. Динамика val/Dice для Soft Dice.\*



!\[val\_loss\_soft](assets/val\_loss\_soft.png)



\*Рис. 5. Динамика val/loss для Soft Dice.\*



!\[predictions\_soft](assets/predictions\_soft.png)



\*Рис. 6. Примеры предсказаний модели с Soft Dice.\*



\### Сравнительный анализ



Обе функции потерь показали отличное качество (`test/Dice > 0.95`). Soft Dice сошелся быстрее по лоссу (val/loss упал до 0.073 против 0.272 у CE), а итоговый `test/Dice` у Soft Dice оказался чуть выше (0.955 против 0.952). Визуально обе модели выдают хорошие маски, но Soft Dice склонен генерировать более гладкие границы объектов.

