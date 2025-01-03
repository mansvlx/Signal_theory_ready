### Обучал модель YOLOv3  
### dataset - [https://huggingface.co/datasets/arnavmahapatra/fruit-detection-dataset](https://huggingface.co/datasets/arnavmahapatra/fruit-detection-dataset)  

* Использовал данные всего датасета,разметку делал через Roboflow

Обучение модели длилось 180 поколений, дальнейшее обучение сочёл не разумным

#### Для обнаружения объектов необходимо ввести в терминал следующую команду:
>***poetry run yolo-detect -m data/config/fruits.cfg -w data/custom_weights/fruits.pth -c data/custom/classes.names -i detection_input -o detection_result***
 
