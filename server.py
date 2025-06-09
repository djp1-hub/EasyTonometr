from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import uuid
import os

from TonometrOSR import TonometerOCR

app = FastAPI()

ocr = TonometerOCR(tessdata_dir="tessdata")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Сохраняем файл во временный путь
        ext = os.path.splitext(file.filename)[1]
        temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ocr_processor = TonometerOCR('tessdata')

        # Запускаем новый процесс распознавания
        # Включаем debug=True, чтобы сохранялись промежуточные изображения
        values = ocr_processor.process_with_7_segment_ocr(temp_filename, debug=True)
        print(values)
        # Обработка изображения
        #values = ocr.process_full(temp_filename)

        # Удаляем временный файл
        os.remove(temp_filename)

        # Формируем ответ
        result = {
            "systolic": values[0],
            "diastolic": values[1],
            "pulse": values[2]
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
