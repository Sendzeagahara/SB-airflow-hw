import os
import dill
import json
import pandas as pd

from datetime import datetime


# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '..')
# path = os.path.expanduser('~/airflow_hw')


def predict():
    model_path = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{model_path[-1]}', 'rb') as file:
        model_pipeline = dill.load(file)

    test_df_preditctions = pd.DataFrame(columns=['id', 'prediction'])
    test_list = os.listdir(f'{path}/data/test')

    for filename in test_list:
        with open(f'{path}/data/test/{filename}', 'r') as test_file:
            form = json.load(test_file)
        test_df = pd.DataFrame.from_dict([form])
        prediction = model_pipeline.predict(test_df)
        pred_df = pd.DataFrame({'id': test_df.id, 'prediction': prediction})
        test_df_preditctions = pd.concat([test_df_preditctions, pred_df])

    test_df_preditctions.to_csv(f'{path}/data/predictions/prediction_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                                index=False)
    pass


if __name__ == '__main__':
    predict()
