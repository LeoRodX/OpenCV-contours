import cv2
import numpy as np
from matplotlib import pyplot as plt

# Чтение изображения
img = cv2.imread('shapes.png')

# Преобразование изображения в градации серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Установка порога для изображения в градациях серого
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Использование функции findContours() для нахождения контуров
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

# Список для хранения названий фигур
for contour in contours:

    # Игнорируем первый контур, потому что findContours распознает все изображение как контур
    if i == 0:
        i = 1
        continue

    # Функция cv2.approxPolyDP() для аппроксимации формы
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # Использование функции drawContours() для рисования контуров
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # Нахождение центра формы
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # Печать названия формы в центре каждой фигуры
    if len(approx) == 3:
        cv2.putText(img, 'triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif len(approx) == 4:
        cv2.putText(img, 'quadrilateral', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif len(approx) == 5:
        cv2.putText(img, 'pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif len(approx) == 6:
        cv2.putText(img, 'hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(img, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Отображение изображения после рисования контуров
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()