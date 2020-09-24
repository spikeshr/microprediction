MY_MUID = '5271a8507f6b4e2b393a7b751577029e'
from microprediction import MicroCrawler
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error




class MyCrawler(MicroCrawler):

    def __init__(self,write_key):
        super().__init__(stop_loss=3.0,min_lags=100,sleep_time=15*60,write_key=write_key,quietude=1,verbose=False)


    def candidate_streams(self):
        bad_names = [] # can use this to exclude troublesome streams
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

    # evaluate a model
    def evaluate_model(self, model_fit, train, test):
    	history = [x for x in train]
    	# make predictions
    	predictions = list()
    	for t in range(len(test)):
    		yhat = model_fit.forecast()[0]
    		predictions.append(yhat)
    		history.append(test[t])
    	# calculate out of sample error
    	error = mean_squared_error(test, predictions)
    	return error

    def sample(self, lagged_values, lagged_times=None, **ignored ):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        uniques = np.unique(lagged_values)
        if len(uniques) < 0.3*len(lagged_values): #arbitrary cutoff of 30% to determine whether outcomes are continuous or quantized
            v = [s for s in (np.random.choice(lagged_values, self.num_predictions))] #randomly select from the lagged values and return as answer
        else:
            """ Exponential Smoothing """
            # evaluate models
            # prepare training dataset
            train_size = int(len(lagged_values) * 0.66)
            train, test = lagged_values[0:train_size], lagged_values[train_size:]
            MSE_list = []
            Overall_fits = []

            fit1 = SimpleExpSmoothing(train, initialization_method="estimated").fit() #SES
            MSE_list.append(self.evaluate_model(fit1, train, test))
            Overall_fits.append(SimpleExpSmoothing(lagged_values, initialization_method="estimated").fit())

            fit2 = Holt(train, initialization_method="estimated").fit() #Holt's
            MSE_list.append(self.evaluate_model(fit2, train, test))
            Overall_fits.append(Holt(lagged_values, initialization_method="estimated").fit())

            fit3 = Holt(train, exponential=True, initialization_method="estimated").fit() #Holt's Exponential
            MSE_list.append(self.evaluate_model(fit3, train, test))
            Overall_fits.append(Holt(lagged_values, exponential=True, initialization_method="estimated").fit())

            fit4 = Holt(train, damped_trend=True, initialization_method="estimated").fit(damping_trend=0.98) #Holt's Additive Damped
            MSE_list.append(self.evaluate_model(fit4, train, test))
            Overall_fits.append(Holt(lagged_values, damped_trend=True, initialization_method="estimated").fit(damping_trend=0.98))

            fit5 = Holt(train, exponential=True, damped_trend=True, initialization_method="estimated").fit() #Holt's Multiplicative Damped
            MSE_list.append(self.evaluate_model(fit5, train, test))
            Overall_fits.append(Holt(lagged_values, exponential=True, damped_trend=True, initialization_method="estimated").fit())

            minpos = MSE_list.index(min(MSE_list))
            model_fit = Overall_fits[minpos]

            point_est,std_err,ci = model_fit.forecast()
            v = [s for s in (np.random.normal(point_est, std_err, self.num_predictions))]

        return sorted(v)



if __name__=="__main__":
    mw = MyCrawler(write_key=MY_MUID)
    mw.set_repository(
        url='https://github.com//spikeshr/microprediction/blob/master/Soggy_Camel.py')
    mw.run(withdraw_all=False)
