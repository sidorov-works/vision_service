# Установка конкретной версии Python в WSL (Ubuntu 24.04)

В этой инструкции используется репозиторий `deadsnakes`, который содержит множество версий Python, недоступных в стандартных репозиториях Ubuntu.

---

## 📋 Подготовка

Откройте терминал WSL (Ubuntu) и выполните первую команду:

```bash
sudo apt update
```

---

## 🔧 Добавление репозитория deadsnakes

### Шаг 1. Добавляем PPA

```bash
echo 'Types: deb
URIs: https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/
Suites: noble
Components: main
' | sudo tee /etc/apt/sources.list.d/deadsnakes-ppa.sources
```

---

### Шаг 2. Добавляем GPG-ключ

```bash
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA6932366A755776
```

> **Примечание:** Если эта команда не сработает, попробуйте вместо неё:
> ```bash
> sudo curl -fsSL https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776 | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg
> ```

---

### Шаг 3. Обновляем список пакетов

```bash
sudo apt update
```

В выводе должны появиться строки, начинающиеся с `Hit: https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu noble InRelease`.

---

## 🐍 Установка Python

### Установите нужную версию

Замените `3.11` на нужную вам версию (например, `3.10`, `3.9` и т.д.):

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```

> **Что установится:**
> * `python3.11` — сам интерпретатор
> * `python3.11-venv` — модуль для создания виртуальных окружений
> * `python3.11-dev` — заголовочные файлы для сборки C-расширений

---

## ✅ Проверка

```bash
python3.11 --version
```

Должно показать: `Python 3.11.x`

---

## 📁 Создание виртуального окружения

Перейдите в папку вашего проекта и создайте изолированное окружение:

```bash
cd ~/projects/мой_проект
python3.11 -m venv venv
```

Активируйте его:

```bash
source venv/bin/activate
```

После активации в начале строки появится `(venv)`.

---

## 📦 Установка пакетов

Обновите `pip` и установите зависимости:

```bash
pip install --upgrade pip
pip install -r requirements.txt   # если есть файл
```

Или установите пакеты вручную:

```bash
pip install numpy pandas torch
```

---

## 🚪 Выход из виртуального окружения

```bash
deactivate
```

---

## 📌 Важные замечания

* **Не удаляйте системный Python** (`python3`). Он нужен для работы системы.
* **Не устанавливайте пакеты глобально** через `pip` без виртуального окружения — это может привести к конфликтам.
* Всегда создавайте отдельное виртуальное окружение для каждого проекта.

---

## 🔄 Если нужно установить другую версию

Повторите шаг **Установка Python**, заменив `3.11` на нужную версию:

```bash
sudo apt install python3.10 python3.10-venv python3.10-dev
```

---

## 🗑️ Удаление версии (если понадобится)

```bash
sudo apt remove python3.11 python3.11-venv python3.11-dev
```

---

## 📚 Полезные команды

| Команда | Что делает |
|---------|------------|
| `python3.11 --version` | Показать версию |
| `which python3.11` | Показать путь к интерпретатору |
| `source venv/bin/activate` | Активировать виртуальное окружение |
| `deactivate` | Выйти из виртуального окружения |
| `pip freeze` | Показать установленные пакеты |
| `pip freeze > requirements.txt` | Сохранить список пакетов в файл |

---

## 🆘 Если что-то пошло не так

Если какая-то команда выдала ошибку — скопируйте текст ошибки и покажите её. В большинстве случаев проблема решается добавлением ещё одного пакета или сменой сервера для ключа.