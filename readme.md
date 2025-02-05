# Kraken
Инструмент для автоматического отбора признаков в задачах машинного обучения
## Tool for var selection
![Example image](images/IMG_20230427_165516_458.jpg)

## Описание
Kraken - это инструмент, который помогает автоматизировать процесс отбора признаков для моделей машинного обучения, работающих с временными рядами. 

Основная идея заключается в последовательном добавлении признаков в модель на основе их важности и влияния на качество предсказаний. 
Отдельно представлен иснтрумент для корректной кросс валидации для многомерных временных рядов.

## Основные возможности
- Автоматический отбор значимых признаков с учетом временной структуры данных
- Поддержка задач регрессии и классификации 
- Совместимость с моделями, поддерживающими scikit-learn API
- Настраиваемые метрики качества
- Возможность использования пользовательских метрик
- Детальное логирование процесса отбора признаков

## Как это работает
1. Сначала инструмент ранжирует все доступные признаки по их важности
2. Затем последовательно добавляет признаки, начиная с самых важных
3. На каждом шаге проверяется, улучшает ли добавление признака качество модели
4. Процесс продолжается, пока добавление новых признаков улучшает результат

## Примеры использования
В репозитории доступны Jupyter ноутбуки с примерами:

### example_regression.ipynb
Демонстрирует использование Kraken для задачи регрессии:
- Создание синтетического датасета с временной структурой
- Настройка кросс-валидации с учетом времени
- Применение LightGBM регрессора
- Отбор признаков с использованием MAPE в качестве метрики

### example_classification.ipynb
Показывает применение для задачи классификации:
- Создание синтетического датасета с временной структурой
- Пример использования кастомной метрики для отбора
- Применение LightGBM классификатора

## Требования
- Python ≥ 3.7
- numpy
- pandas
- scikit-learn
- shap
- lightgbm

