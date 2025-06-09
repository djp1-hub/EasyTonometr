import cv2
import numpy as np
import pytesseract
import os

class TonometerOCR:
    def __init__(self, tessdata_dir, lang="ssd", psm=7):
        self.tessdata_dir = tessdata_dir
        self.lang = lang
        self.psm = psm



    def resize_to_width(self, image, width=100):
        h, w = image.shape[:2]
        scale = width / w

        new_dim = (width, int(h * scale))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)

    def invert(self, image):
        return cv2.bitwise_not(image)


    def morf_closed(self, image):
        img = cv2.imread("lcd_rectified.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # === 2. Морфология для склейки сегментов в цифры ===
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))  # вертикально удлинённое ядро
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imwrite("debug_morph_closed.jpg", closed)
        return closed


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

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        cv2.imwrite("lcd_rectified.jpg", warped)
        print("[INFO] LCD дисплей выровнен и сохранён: lcd_rectified.jpg")
        return warped

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_values_from_lcd(self, lcd_img):
        gray = cv2.cvtColor(lcd_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        row1 = gray[int(0.20*h):int(0.45*h),  int(0.40 * w):w-int(0.09 * w)]
        row2 = gray[int(0.45*h):int(0.7*h),   int(0.50 * w):w-int(0.09 * w)]
        row3 = gray[int(0.7*h):int(0.90*h),   int(0.59 * w):w-int(0.085 * w)]
        rows = [row1, row2, row3]

        results = []
        for i, row in enumerate(rows):
            blur = cv2.GaussianBlur(row, (0, 0), sigmaX=10, sigmaY=10)
            norm = cv2.addWeighted(row, 1.5, blur, -0.5, 0)
            resize = self.resize_to_width(norm, 65)

            filename = f"ocr_row_{i+1}.jpg"
            cv2.imwrite(filename, resize)
            print(f"[DEBUG] Сохранил {filename}")

            config = f"--psm {self.psm} -l {self.lang} --tessdata-dir {self.tessdata_dir}"
            text = pytesseract.image_to_string(resize, config=config)

            digits = ''.join(filter(str.isdigit, text))
            print(f"[OCR] Row {i+1}: '{text.strip()}' → '{digits}'")
            results.append(digits if digits else None)

        return results

    def process(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        lcd = self.extract_lcd_screen(image)
        # cv2.imwrite("lcd.jpg", lcd)

        values = self.extract_values_from_lcd(lcd)
        if len(values) == 3:
            print(f"[RESULT] SYS: {values[0]}, DIA: {values[1]}, PULSE: {values[2]}")
        else:
            print("[WARN] Не все значения получены:", values)
        return values
    def process_full(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        lcd = self.extract_lcd_screen(image)
        morf_closed = self.morf_closed(lcd)
        config = f"--psm 3 -l ssd --tessdata-dir tessdata"
        text = pytesseract.image_to_string(morf_closed, config=config)
        values = text.strip().splitlines()
        print(values)
        return values




