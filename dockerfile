# Используем официальный образ Python
FROM python:3.8-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3-dev \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y g++ 
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y swig
RUN pip install --upgrade pip
RUN pip install gymnasium[box2d]
     
# Копируем файлы проекта в рабочую директорию контейнера
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Определяем команду для запуска кода
CMD ["python", "main.py"]