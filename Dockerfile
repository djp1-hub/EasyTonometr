FROM python:3.11-slim

# Установка системных зависимостей: Tesseract + библиотеки для OpenCV и EasyOCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей и установка Python-библиотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Запуск FastAPI через uvicorn
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
