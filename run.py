from keras import Sequential
from keras.layers import LSTM,Dropout,Dense,Activation
import pandas as pd
import joblib
import numpy as np
y2p = joblib.load("year2production")
anyang = pd.read_csv("anyang.csv",index_col=0)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class WheatProduction:
    def __init__(self):
        self.scaler = StandardScaler()
        self.y_scaler = MinMaxScaler()


    def apply_transform_item(self,item:np.ndarray):
        result = []
        for k in range(item.shape[2]):
            if not self.scalers.get(k):
                self.scalers[k] = StandardScaler()
            result.append(self.scalers[k].fit_transform(item[:,:,k].reshape((-1,1))).reshape(1,7,1))
        return np.stack(result)

    def apply_transform(self, item: pd.DataFrame):
        ix = item.drop(["year"],axis=1)
        result = pd.DataFrame(self.scaler.fit_transform(ix.values),columns=item.columns[1:])
        result["year"] = item["year"]
        return result


    def builder(self,df:pd.DataFrame):
        trainx,trainy = [],[]
        df = self.apply_transform(df)
        for g,item in df.groupby(by=["year"]):
            item.drop(["year"],inplace=True,axis=1)
            x = np.array(item.values)
            x = x[np.newaxis,:,:]
            # self.apply_transform(x)
            trainy.append(y2p[g])
            trainx.append(x)
        return np.concatenate(trainx, axis=0),np.array(trainy).reshape([-1,1])

    def model_init(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(7, 5), return_sequences=False))
        model.add(Dense(64))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        return model

    def y_transform(self,y:np.ndarray):
        return self.y_scaler.fit_transform(y)

    def train(self):
        X,Y = self.builder(anyang)
        y_transed = self.y_transform(Y)
        model = self.model_init()
        model.fit(X,y_transed,epochs=10000)
        predits = model.predict(X[0:9, :, :])
        print(self.y_scaler.inverse_transform(predits))
        print(Y)
        joblib.dump(model,"model")

    def load_model(self):
        self.model = joblib.load("model")

    def evaluate(self):
        self.load_model()
        X, Y = self.builder(anyang)
        self.y_scaler.fit(Y)
        predicts = self.model.predict(X)
        predicts = self.y_scaler.inverse_transform(predicts)
        df = pd.DataFrame({"label":Y.reshape((-1)),"predi":predicts.reshape((-1))},index=range(Y.shape[0]))
        df.to_csv("test.csv")


model = WheatProduction()
model.evaluate()

