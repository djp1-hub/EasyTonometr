import cv2
import numpy as np
from imutils import contours
import os
import shutil


# Вам может понадобиться pytesseract для старых методов, если вы их используете
# import pytesseract

class TonometerOCR:
    # Конструктор остался прежним, хотя для нового метода он не используется
    def __init__(self, tessdata_dir=None, lang="ssd", psm=7):
        self.tessdata_dir = tessdata_dir
        self.lang = lang
        self.psm = psm

    # --- Ваши существующие вспомогательные методы (без изменений) ---
    def resize_to_width(self, image, width=100):
        h, w = image.shape[:2]
        scale = width / w
        new_dim = (width, int(h * scale))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)

    def invert(self, image):
        return cv2.bitwise_not(image)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_lcd_screen(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lcd_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            # Условие можно подбирать под ваши изображения
            if area > 50000 and len(approx) == 4:
                if area > max_area:
                    lcd_cnt = approx
                    max_area = area

        if lcd_cnt is None:
            raise ValueError("LCD дисплей не найден")

        pts = lcd_cnt.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        print("[INFO] LCD дисплей выровнен.")
        return warped

    # --- Ваш метод morf_closed, немного улучшенный ---
    def morf_closed(self, rectified_img):
        gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Сохраняем для отладки, как и раньше
        cv2.imwrite("debug_morph_closed.jpg", closed)
        return closed

    # +++ НАШ НОВЫЙ ВСТРОЕННЫЙ МЕТОД РАСПОЗНАВАНИЯ +++
    def recognize_7_segment(self, binary_image, debug=False, debug_dir="debug_output"):
        # Шаблоны цифр
        DIGITS_TEMPLATES = {
            "1": (0, 1, 1, 0, 0, 0, 0), "2": (1, 1, 0, 1, 1, 0, 1),
            "3": (1, 1, 1, 1, 0, 0, 1), "4": (0, 1, 1, 0, 0, 1, 1),
            "5": (1, 0, 1, 1, 0, 1, 1), "6": (1, 0, 1, 1, 1, 1, 1),
            "7": (1, 1, 1, 0, 0, 0, 0), "8": (1, 1, 1, 1, 1, 1, 1),
            "9": (1, 1, 1, 1, 0, 1, 1), "0": (1, 1, 1, 1, 1, 1, 0)
        }

        # Для отладки нам нужно 3-канальное изображение, чтобы рисовать цветные рамки
        image_for_debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        h_img, w_img, _ = image_for_debug.shape

        # Эрозия и поиск контуров
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        cnts = cv2.findContours(eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if not cnts: return []

        # Фильтрация контуров
        digit_cnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w < (w_img * 0.8) and h < (h_img * 0.8) and w > 5 and h > (h_img * 0.1):
                digit_cnts.append(c)

        if not digit_cnts: return []

        if debug:
            img_with_filtered = image_for_debug.copy()
            for c in digit_cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(img_with_filtered, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, "02_filtered_contours.png"), img_with_filtered)

        # Сортировка по строкам
        (sorted_cnts, _) = contours.sort_contours(digit_cnts, method="top-to-bottom")
        rows = []
        if sorted_cnts:
            current_row = [sorted_cnts[0]]
            for i in range(1, len(sorted_cnts)):
                (x, y, w, h) = cv2.boundingRect(sorted_cnts[i])
                (x_prev, y_prev, w_prev, h_prev) = cv2.boundingRect(current_row[0])
                if abs(y - y_prev) < h_prev * 0.5:
                    current_row.append(sorted_cnts[i])
                else:
                    rows.append(current_row)
                    current_row = [sorted_cnts[i]]
            rows.append(current_row)

        # Распознавание и сбор результата
        all_rows_text = []
        for i, row in enumerate(rows):
            (row_sorted, _) = contours.sort_contours(row, method="left-to-right")
            current_row_text = ""
            for c in row_sorted:
                (x, y, w, h) = cv2.boundingRect(c)
                digit = "?"
                aspect_ratio = h / float(w) if w > 0 else 0
                if aspect_ratio > 5.0:
                    digit = "1"
                else:
                    roi = binary_image[y:y + h, x:x + w]
                    # ... (остальная логика распознавания сегментов)
                    (roi_h, roi_w) = roi.shape
                    (dW, dH) = (int(roi_w * 0.3), int(roi_h * 0.15))
                    dHC = int(roi_h * 0.05)
                    segments = [
                        ((0, 0), (w, dH)), ((w - dW, 0), (w, h // 2)), ((w - dW, h // 2), (w, h)),
                        ((0, h - dH), (w, h)), ((0, h // 2), (dW, h)), ((0, 0), (dW, h // 2)),
                        ((w // 2 - dW // 2, h // 2 - dHC), (w // 2 + dW // 2, h // 2 + dHC))
                    ]
                    on = [0] * len(segments)
                    for j, ((xA, yA), (xB, yB)) in enumerate(segments):
                        seg_roi = roi[yA:yB, xA:xB]
                        total = cv2.countNonZero(seg_roi)
                        area = (xB - xA) * (yB - yA)
                        if area > 0 and total / float(area) > 0.4: on[j] = 1
                    try:
                        digit = list(DIGITS_TEMPLATES.keys())[list(DIGITS_TEMPLATES.values()).index(tuple(on))]
                    except ValueError:
                        pass
                current_row_text += digit
            all_rows_text.append(current_row_text)
        return all_rows_text

    # +++ НОВЫЙ ГЛАВНЫЙ МЕТОД ДЛЯ ЗАПУСКА +++
    def process_with_7_segment_ocr(self, image_path, debug=False):
        # Подготовка папки для отладки
        if debug:
            debug_dir = "debug_output"
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)
            os.makedirs(debug_dir)

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

            # Шаг 1: Найти и выровнять экран
            lcd_image = self.extract_lcd_screen(image)

            # Шаг 2: Применить морфологию для получения бинарного изображения
            binary_lcd = self.morf_closed(lcd_image)

            # Шаг 3: Распознать цифры на бинарном изображении
            values = self.recognize_7_segment(binary_lcd, debug=debug)

            return values

        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] Ошибка обработки: {e}")
            return None


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ КЛАССА ---
if __name__ == '__main__':
    # Путь к вашему изображению тонометра
    image_file_path = 'IMG_2033 2 Medium.jpeg'  # ЗАМЕНИТЕ НА ВАШ ФАЙЛ

    # Создаем экземпляр нашего класса
    ocr_processor = TonometerOCR()

    # Запускаем новый процесс распознавания
    # Включаем debug=True, чтобы сохранялись промежуточные изображения
    results = ocr_processor.process_with_7_segment_ocr(image_file_path, debug=True)
    print(results)

