MY_MUID = 'INSERT YOUR KEY HERE'
from microprediction import MicroCrawler
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order




class MyCrawler(MicroCrawler):

    def __init__(self,write_key):
        super().__init__(stop_loss=3.0,min_lags=50,sleep_time=15*60,write_key=write_key,quietude=1,verbose=False)


    def candidate_streams(self):
        bad_names = ['electricity','emojitracker'] # can use this to exclude troublesome streams
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
        if len(uniques) < 0.2*len(lagged_values): #arbitrary cutoff of 20% to determine whether outcomes are continuous or quantized
            v = [s for s in (np.random.choice(lagged_values, self.num_predictions))] #randomly select from the lagged values and return as answer
        else:
            rev_values = lagged_values[::-1] # our data are in reverse order, the AR model needs the opposite
            """ Simple Autoregression """
            ARmodel = ar_select_order(rev_values, maxlag=int(0.1*len(rev_values)))
            model_fit = ARmodel.model.fit()
            point_est = model_fit.predict(start=len(rev_values), end=len(rev_values), dynamic=False)
            st_dev = np.std(rev_values)
            #v = [s for s in (np.random.normal(point_est, st_dev, self.num_predictions))]
            v = [s for s in (np.linspace(start=point_est-2*st_dev,stop=point_est+2*st_dev, num=self.num_predictions))] #spread the predictions out evenly
            print (*v, sep = ", ")
        return v


if __name__=="__main__":
    mw = MyCrawler(write_key=MY_MUID)
    mw.set_repository(
        url='https://github.com//spikeshr/microprediction/blob/master/Fledge_Goat.py')
    mw.run(withdraw_all=True)
