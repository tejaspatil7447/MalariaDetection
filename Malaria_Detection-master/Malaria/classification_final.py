#	#	#	#	#	#	#	#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import metrics
import pickle

#	#	#	#	#	#	#	#

df = pd.read_csv("csv/Dataset.csv")
#print(df.head(2))

#	#	#	#	#	#	#	#

x = df.drop(['Label'],axis=1)
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#	#	#	#	#	#	#	#

#model = RandomForestClassifier(n_estimators=100,max_depth=5)
#model.fit(x_train,y_train)
#pickle.dump(model,open("finalized_model.pkl","wb"))

#	#	#	#	#	#	#	#

model = pickle.load(open('finalized_model.pkl', 'rb'))

#	#	#	#	#	#	#	#

predictions = model.predict(x_test)

print(metrics.classification_report(predictions,y_test))

#	#	#	#	#	#	#	#