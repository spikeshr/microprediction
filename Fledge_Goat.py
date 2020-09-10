MY_MUID = 'INSERT YOUR KEY HERE'
from microprediction import MicroCrawler
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order




class MyCrawler(MicroCrawler):

    def __init__(self,write_key):
        super().__init__(stop_loss=3.0,min_lags=50,sleep_time=15*60,write_key=write_key,quietude=1,verbose=False)


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

    def sample(self, lagged_values, lagged_times=None, **ignored ):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        uniques = np.unique(lagged_values)
        if len(uniques) < 0.3*len(lagged_values): #arbitrary cutoff of 30% to determine whether outcomes are continuous or quantized
            v = [s for s in (np.random.choice(lagged_values, self.num_predictions))] #randomly select from the lagged values and return as answer
        else:

            """ Simple Autoregression """
            ARmodel = ar_select_order(lagged_values, maxlag=int(0.1*len(lagged_values)))
            model_fit = ARmodel.model.fit()
            point_est = model_fit.predict(start=len(lagged_values), end=len(lagged_values), dynamic=False)
            st_dev = np.std(lagged_values)
            v = [s for s in (np.random.normal(point_est, st_dev, self.num_predictions))]

        return sorted(v)


if __name__=="__main__":
    mw = MyCrawler(write_key=MY_MUID)
    mw.set_repository(
        url='https://github.com//spikeshr/microprediction/blob/master/Fledge_Goat.py')
    mw.run(withdraw_all=False)
