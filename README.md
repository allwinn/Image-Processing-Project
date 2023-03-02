# Image-Processing-Project
This repository cotains code for image processing exam tasks

# Contributors
Ameerali Khan (220200827)
Allwin Noble (220200654)

## Train and evaluation.
Each tasks can be train and evaluated seperatedly. 

```python -m src.train_evaluate -t [Task_Name] {--train, --predict} -m [Model_Name] {--classical_model [CLASSICAL_MODEL]}```

1. `-t` Task_Name supports following values:
    * Classification
    * Segmentation
    * Deskewing
    * Cleaning
    * OCR

2. By default script evaluate the given task. In this case `-m` model name should be passed.
3. `--train` flag should be passed for training. script generates the model name base on the configuration.
4. Once model is trained evaluation and prediction can be done by using `--predict` flag. Here `-m` model name is required parameter.
5. Finally `--classical_model` parameter is need only for deskewing task. For now deskewing task only support `houg_transform`.
