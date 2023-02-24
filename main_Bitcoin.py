import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib

import requests

app = Flask(__name__)



# Actual是正確答案
include=['Low', 'High', 'Open', 'Close', 'Volume', "Mean", 'Actual']
# 依賴變量是我們的答案?Actual
dependent_variable = include[-1]


model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
# 預測時填充
model_columns = None
clf = None

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello world!!"


@app.route('/predict', methods=['POST'])
def predict():
    # 傳入模型讓他訓練就好了
    if clf:
        try:
            # 傳入 Low High Open Close Volume Mean
            # 給他傳入json檔案(在調用網頁時)當作test_X
            json_ = request.json
            # 將X_test轉換成DataFrame類型
            # Low   High   Open  Close  Volume   Mean
            # 0  10000  20000  15000  11000    3000  15000
            query = pd.DataFrame(json_)
            
            
            # Age  Sex_female  Sex_male  Embarked_C  Embarked_S
        # 0   85           0         1           0           1
        # 1   24           1         0           1           0
        

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori

            # 資料標準化
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            num_pl = make_pipeline(SimpleImputer(strategy='mean'), 
                            StandardScaler())
            query = num_pl.fit_transform(query)
            query = pd.DataFrame(query)
            print(query)
            
            return "OK"

            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = list(clf.predict(query))

            # Converting to int from int64
            # map(int,prediction) ->把prediction裡面的內容變成int類型?再放到list當中
            # 如果有小數點他就會無條件捨去變成整數
            return jsonify({"prediction": list(map(int, prediction))})
        # 有錯誤發生 傳回錯誤json格式
        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    # 還沒有訓練模型
    else:
        print('train first')
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    
# inputs
    url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    param = {"convert":"USD","slug":"bitcoin","time_end":"1658275200","time_start":"1367107200"}
    content = requests.get(url=url, params=param).json()
    df = pd.json_normalize(content['data']['quotes'])

    # Extracting and renaming the important variables
    df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
    df['Low'] = df['quote.USD.low']
    df['High'] = df['quote.USD.high']
    df['Open'] = df['quote.USD.open']
    df['Close'] = df['quote.USD.close']
    df['Volume'] = df['quote.USD.volume']

    # Drop original and redundant columns
    df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

    # Creating a new feature for better representing day-wise values
    df['Mean'] = (df['Low'] + df['High'])/2

    # Cleaning the data for any NaN or Null fields
    df = df.dropna()

    # Creating a copy for making small changes
    dataset_for_prediction = df.copy()
    dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift(-1)
    dataset_for_prediction=dataset_for_prediction.dropna()

    # date time typecast
    dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
    dataset_for_prediction.index= dataset_for_prediction['Date']
    # index已經設為Date了 Date column可以丟棄
    dataset_for_prediction.drop('Date',axis=1,inplace=True)
    
    # 資料標準化
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    num_pl = make_pipeline(SimpleImputer(strategy='mean'), 
                       StandardScaler())
    dataset_for_prediction_pl = num_pl.fit_transform(dataset_for_prediction)
    # 把它做成資訊框
    dataset_for_prediction_pl= pd.DataFrame(dataset_for_prediction_pl)
    # 重新命名
    dataset_for_prediction_pl.rename(columns={0:"Low", 1:"High", 2:"Open", 3:"Close", 4:"Volume", 5:"Mean", 6:"Actual"}, inplace=True)
    # X 為 Low High Open Close Volume Mean
    # Y 為 Actual
    # 把Actual取出 其他為X
    # Sort =False防止Column的index改變
    X = dataset_for_prediction_pl[dataset_for_prediction_pl.columns.difference([dependent_variable], sort=False)]
    y = dataset_for_prediction_pl[dependent_variable]

    

    # capture a list of columns that will be used for prediction
    global model_columns
    # 把 Actual拿掉 ，也就是答案
    model_columns = include[:-1]
    
    
    # 把model 的column也就是x變成 pkl檔案放在資料夾中(model_columns_file_name)
    # pkl是常見的讀取模型檔案
    joblib.dump(model_columns, model_columns_file_name)

    # 引入算法 
    from sklearn import linear_model
    global alg
    alg = linear_model.BayesianRidge()
    start = time.time()
    alg.fit(X, y)
    # 保存我們的模型
    joblib.dump(alg, model_file_name)
    

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % alg.score(X, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
    return return_message


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        # 第一個參數
        port = int(sys.argv[1])
    except Exception as e:
        port = 80


    try:
        clf = joblib.load(model_file_name)
        # 假如有model
        # ['Age' 'Embarked_C' 'Embarked_Q' 'Embarked_S' 'Embarked_nan' 'Sex_female'  'Sex_male' 'Sex_nan']
        print('model loaded')
        # 載入模型index
        # ['Age', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan', 'Sex_female', 'Sex_male', 'Sex_nan']
        # model_column.pkl
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')


# 找不到所以是第一次創建?
# 錯誤擷取 e是程式錯誤訊息
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
