import pandas as pd
import numpy as np
from numpy.core.defchararray import isdigit
import joblib
df = pd.read_excel("anyang.xlsx",index_col=False)

location,label = [],[]
temp = []
rain = []
rediation = []
ndvi = []
evi = []
production = {}
result = []
def trans(x):
    global temp,location,rediation,rain,result,label
    if isinstance(x[0],str): #and isdigit(x[0].split('-')[0]):
        print(x[0])
        production.update({x[0]:x[34]})
        label.extend([x[0]]*7)
        temp.extend(x[1:8].values)
        rain.extend(x[8:15].values)
        rediation.extend(x[15:22].values)
        ndvi.extend(list(np.concatenate(((0,),x[22:28].values))))
        evi.extend(list(np.concatenate(((0,), x[28:34].values))))

df.apply(trans,axis=1)
df2 = pd.DataFrame({"year":label,"temp":temp,"rain":rain,"rediation":rediation,"evi":evi,"ndvi":ndvi})
df2.to_csv("anyang.csv")
print(production)
joblib.dump(production,"year2production")
print(df2)






