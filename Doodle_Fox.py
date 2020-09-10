MY_MUID = 'YOUR MUID KEY' #update this yourself
from microprediction import MicroCrawler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error




class MyCrawler(MicroCrawler):

    def __init__(self,write_key):
        super().__init__(stop_loss=3.0,min_lags=50,sleep_time=15*60,write_key=write_key,quietude=1,verbose=False)


    def candidate_streams(self): # can use this to exclude troublesome streams, or just delete
        bad_names = [] # enter some portion of the names of streams you want to exclude here
        good_names = []
        candidate_names = [name for name, sponsor in self.get_streams_by_sponsor().items()] # if name[:1] != 'z' #exclude z streams


        for cname in candidate_names:
            okay = True
            for bname in bad_names:
                if bname in cname:
                    okay = False
                    break
            if okay == True:
                good_names.append(cname)
        return good_names

    # evaluate an ARIMA model for a given order (p,d,q)
    def evaluate_arima_model(self, X, arima_order):
    	# prepare training dataset
    	train_size = int(len(X) * 0.66)
    	train, test = X[0:train_size], X[train_size:]
    	history = [x for x in train]
    	# make predictions
    	predictions = list()
    	for t in range(len(test)):
    		model = ARIMA(history, order=arima_order)
    		model_fit = model.fit(disp=0)
    		yhat = model_fit.forecast()[0]
    		predictions.append(yhat)
    		history.append(test[t])
    	# calculate out of sample error
    	error = mean_squared_error(test, predictions)
    	return error

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(self, dataset, p_values, d_values, q_values):
    	#dataset = dataset.astype('float32')
    	best_score, best_cfg = float("inf"), (0,0,0)
    	for p in p_values:
    		for d in d_values:
    			for q in q_values:
    				order = (p,d,q)
    				try:
    					mse = self.evaluate_arima_model(dataset, order)
    					if mse < best_score:
    						best_score, best_cfg = mse, order
    				except:
    					continue
    	return best_cfg

    def sample(self, lagged_values, lagged_times=None, **ignored ):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        uniques = np.unique(lagged_values)
        if len(uniques) < 0.3*len(lagged_values): #arbitrary cutoff of 30% to determine whether outcomes are continuous or quantized
            v = [s for s in (np.random.choice(lagged_values, self.num_predictions))] #randomly select from the lagged values and return as answer
        else:

            """ Simple ARIMA """
                # evaluate parameters
            p_values = [0, 1, 2, 4, 6, 8, 10] #these are kind of arbitrary, but need to put a limit on it
            d_values = range(0, 3) #arbitrary, but need to put a limit on it
            q_values = range(0, 3) #arbitrary, but need to put a limit on it
            best_order = self.evaluate_models(lagged_values, p_values, d_values, q_values)
            arma_mod = ARIMA(lagged_values, order=best_order, trend='n')
            model_fit = arma_mod.fit()
            point_est = model_fit.predict(len(lagged_values), len(lagged_values), dynamic=True)
            st_dev = np.std(lagged_values)
            v = [s for s in (np.random.normal(point_est, st_dev, self.num_predictions))]

        return sorted(v)



if __name__=="__main__":
    mw = MyCrawler(write_key=MY_MUID)
    mw.set_repository(
        url='https://github.com//spikeshr/microprediction/blob/master/Doodle_Fox.py')
    mw.run(withdraw_all=False)
