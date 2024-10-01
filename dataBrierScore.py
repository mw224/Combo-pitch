# residualBrierScore.py

def residualBrierScore(p,df_merged):
    
    # Third-party imports
    import pandas as pd
    
    # Calculate the second summation in the Brier score formula
    n = len(df_merged)
    E = pd.DataFrame([0] * n, index=df_merged.index)
    E.rename(columns={0:'event_occurred'}, inplace=True)
    E.at[p,'event_occurred'] = 1
    resa = df_merged - E.values
    resb = resa.pow(2)
    res = resb.sum(min_count=1).to_frame()
    
    # Output
    return res




    #    P = 1/n     sum _ j to r        sum_ i to n     (f_ij - E_ij)^2
     #       P is the score
      #      r is the number of possible classes
       #     n is the number of occasions (i.e., number of forecasts)
        #    f is the forecast probabilities that the event will occur in each class
         #   E takes the value 1 or 0 according to whether the event occurred in class j or not.

