import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from models.statistical.arima import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW


class Holy(ARIMA):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)
        
        # self.period = opts.period
        # self.trend = opts.trend
        # self.seasonal = opts.seasonal
    
    def gen_model(self, ):
        # self.model = HW(self.history, trend =  self.trend, seasonal=self.seasonal,seasonal_periods=self.period).fit()
        self.model = HW(self.history,seasonal_periods=self.params.period ).fit()        
    
    def _predict(self, H):
        pred = self.model.forecast(H)
        return pred
          
    # def predict(self, input):
    #     H = self.params.H
        
    #     test_samples = input.shape[0]
    #     pred_sec = H + input.shape[0] - 1
        
    #     if self.refit == False:
    #         self.gen_model()
            
    #         yPred = self.model.forecast(pred_sec) - self.constant
    #         yPred = create_dataset(yPred, H - 1)
            
    #     else:
    #         yPred = np.empty((test_samples, H))
    #         for i in trange(test_samples):
    #             self.model = HW(self.history, trend =  self.trend, seasonal=self.seasonal,seasonal_periods=self.period).fit()
                
    #             yPred[i,:]= self.model.forecast(H) - self.constant
    #             self.history = np.concatenate((self.history, input[i,-1].reshape(-1,)), axis=0)
    #             self.check_history()
    #             # print(history[-3:])
    #     return yPred