### Запуск в Google Colab
Обучение можно запустить в [Google Colab](https://colab.research.google.com/) с бесплатным T4.


Так как ваш репозиторий приватный (создаётся через GitHub Classroom), вам нужен [Personal Access Token (PAT)](https://github.com/settings/tokens) со scope `repo`.

1. Идём на https://github.com/settings/tokens → Generate new token (classic)
<img src="assets/access-token-1.png" alt="access token settings" height="150">

2. Задаём имя токена, устанавливаем срок жизни (на время выполненя проекта), выбираем scope = `repo`
<img src="assets/access-token-2.png" alt="access token scope" height="300">

3. Отматываем вниз страницы, жмём Создать. Копируем получившийся токен куда-нибудь в текстовый файл
4. Идём в Colab, жмём иконку 🔑 (левый сайдбар) → Add new secret
    - Name: GH_TOKEN
    - Value: токен из шага 3
    - Включаем "Notebook access"

    <img src="assets/access-token-3.png" alt="access token colab" height="300">

Затем копируете в новый ноутбук следующие ячейки:

```python
# 1) Авторизация и клонирование репозитория
from google.colab import userdata
GH_TOKEN = userdata.get("GH_TOKEN")
GH_USER  = "ML-BI-25"   # название аккаунта, в нашем случае это всегда ML-BI-25
REPO     = "cell-segmentation-norsage" # cell-segmentation-<username>
!git clone https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO}.git
%cd {REPO}
```

```bash
# 2) Установка зависимостей под GPU
!pip install -q -r requirements-linux-cuda.txt
```
> После установки вы можете увидеть предупреждения вроде таких:
> ```
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
> torchaudio 2.10.0+cu128 requires torch==2.10.0, but you have torch 2.8.0 which is incompatible.
> tensorflow 2.19.0 requires tensorboard~=2.19.0, but you have tensorboard 2.20.0 which is incompatible.
> jupyter-kernel-gateway 2.5.2 requires notebook<7.0,>=5.7.6, but you have notebook 7.3.3 which is incompatible.
> ```
> Это не страшно, они касаются других заранее установленных зависимостей в Colab, нам это мешать не должно.

```bash
# 3) Скачиваем и распаковываем датасет
!mkdir data && cd data && wget -q https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3799/yeast_cell_in_microstructures_dataset.zip && unzip -q yeast_cell_in_microstructures_dataset.zip -d yeast_cell_in_microstructures_dataset
```

```bash
# 4) Запускаем обучение
!python -m scripts.train
```

```python
# 5) Архивируем и скачиваем логи и чекпоинты
!zip -qr logs.zip logs/
from google.colab import files
files.download("logs.zip")
```

После скачивания распакуйте `logs.zip` в корень локального проекта и откройте Tensorboard локально (`tensorboard --logdir logs/`).

> **Внимание:** после отключения сессии все данные из окружения Colab удаляются. Поэтому все шаги нужно повторять каждый раз заново. Также не забывайте коммитить и пушить изменения кода в свой репозиторий **до** запуска Colab — иначе они до Colab не доедут.

Пока окружение активно, новую версию кода можно стягивать командой `!git pull` из любо ячейки Colab ноутбука.
