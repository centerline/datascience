from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np 
from sklearn.externals import joblib

def main():
	print "start loading data..." 
	traindata = pd.read_csv("./ato135_dev_woe_tm.csv.test")
	#traindata = pd.read_csv("./ato135_dev_woe_tm.csv")
	print "done loading" 

	feature_names = list(traindata.columns)
	feature_names.remove("trans_id")   
	feature_names.remove("ato_label")   
	feature_names.remove("unit_wgt")  
	feature_names.remove("dollar_wgt")   
	features = traindata[feature_names].values
	target = traindata["ato_label"].values

	features = np.asarray(features, dtype=np.float32, order='F')
	filename = 'test.joblib'
	print "put data in the right layout and map to " + filename
	joblib.dump(features, filename)
	features = joblib.load(filename, mmap_mode='c')

	print "start fit() ...." 
	rf = RandomForestClassifier(verbose=1, n_estimators=100, n_jobs=4, criterion='gini', max_depth=None, bootstrap=False, min_samples_split=2,random_state=0)
	rf.fit(features, target)
	print "done fit() ...." 

	print "predicting ...." 
        np.savetxt('./rf_out2.csv', rf.predict_proba(features), delimiter=',', fmt='%f')



if __name__ == "__main__":
	main()

	pass
