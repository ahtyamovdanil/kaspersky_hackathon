# Проект по распознаванию вредоносных url адресов
Было создано две модели на основе catboost и bidirectional GRU.
Рекурентная нейронная сеть показала себя лучше градиентного бустинга, а ансамбль из этих двух мделей давал результат хуже, чем у GRU, поэтому в финале использовалась только GRU.
Для демонстрации результатов работы модели создан телеграм бот **@good_bad_URL_bot** , который определяет вредоносная ли присланная ему ссылка. 

В jupyter ноутбуках **train_catboost** и **train_GRU** код с тренировкой и результатами модели градиентного бустинга и рекурентной сети. 

В **ensemble** результаты работы ансамбля из этих двух моделей
