# ИПР-22-1Б Стрельников Максим

## Описание задачи
Поставил задачу дообучить модель YOLO на заданном датасете для определения фруктов (банан, яблоко, апельсин).

## Процесс работы
Для составления описаний изображений:
1. Использовал сервис Roboflow.
2. Процесс разметки оказался трудоемким, так как задача была учебной и не предполагала высокой точности.

### Разделение датасета
- Разделение датасета было автоматически созданно сервисом(70% train, 20% validation, 10% test).
- Использовал полный объем исходного датасета.

## Валидация
После обучения модели провел валидацию на:
- Фото.

### Результаты
- Результаты вышли не точными, так как исходных данных было мало.

## Полезные ссылки
- [Использованный датасет](https://huggingface.co/datasets/arnavmahapatra/fruit-detection-dataset)
---------------------------
## Примечание
**Для обнаружения объектов необходимо произвести следующие действия:**
1. **Запускать проект на python3.10**
2. **Произвести установку зависимостей с помощью команды poetry install**
