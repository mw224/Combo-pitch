# comboModel.py

############################################################################################################################
# Equal-weight
def comboModel(df_merged,BS,unique_pitches,game_date_i,oos_date,runner_on):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Prepare PDF
    if game_date_i >= oos_date:
        
        #################### EQUAL-WEIGHT
        # Combine (equal weight)
        mComboEWOut = df_merged.mean(axis=1).to_frame()
        mComboEWOut.rename(columns={0:'PDF_comboEW'}, inplace=True)
        
        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboEWOut.at['PO','PDF_comboEW'] = 0
        
        # Ensure final weights sum to 1
        mComboEWOut = mComboEWOut / mComboEWOut.sum()
        
        
        
        #################### EQUAL-WEIGHT of all models with errors better than benchmark
        criteria1 = BS <= BS.iloc[0]        # iloc is 0 because that is model 1, which is the global average benchmark
        criteria2 = df_merged.isnull().all(0) == False
        criteria = np.logical_and(criteria1, criteria2)
        df_merged_red = df_merged[criteria.index[criteria]]
        mComboEW2Out = df_merged_red.mean(axis=1).to_frame()
        mComboEW2Out.rename(columns={0:'PDF_comboEW2'}, inplace=True)
        
        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboEW2Out.at['PO','PDF_comboEW2'] = 0
            
        # Ensure final weights sum to 1
        mComboEW2Out = mComboEW2Out / mComboEW2Out.sum()
        
        
        
        
        #################### INVERSE WEIGHT
        w = (BS / BS.iloc[0]).to_frame()
        w = w.T
        w = w[criteria.index[criteria]]
        w = 1 / w
        w = w.div(w.sum(axis=1), axis=0)
        
        hold = np.dot(df_merged_red.fillna(0),w.T)
        mComboIWOut = pd.DataFrame(hold, index=df_merged_red.index, columns=['PDF_comboIW'])

        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboIWOut.at['PO','PDF_comboIW'] = 0
            
        # Ensure final weights sum to 1
        mComboIWOut = mComboIWOut / mComboIWOut.sum()
        
        # Clear for next models
        del hold
        
        
        
        
        #################### INVERSE WEIGHT with weights squared
        w2 = pow(w,2)
        w2 = w2.div(w2.sum(axis=1), axis=0)
        hold = np.dot(df_merged_red.fillna(0),w2.T)
        mComboIW2Out = pd.DataFrame(hold, index=df_merged_red.index, columns=['PDF_comboIW2'])

        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboIW2Out.at['PO','PDF_comboIW2'] = 0
            
        # Ensure final weights sum to 1
        mComboIW2Out = mComboIW2Out / mComboIW2Out.sum()
        
        # Clear for next models
        del hold, w2
        
        
        
        
        #################### INVERSE WEIGHT with weights to 5th power
        w3 = pow(w,5)
        w3 = w3.div(w3.sum(axis=1), axis=0)
        hold = np.dot(df_merged_red.fillna(0),w3.T)
        mComboIW3Out = pd.DataFrame(hold, index=df_merged_red.index, columns=['PDF_comboIW3'])

        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboIW3Out.at['PO','PDF_comboIW3'] = 0
            
        # Ensure final weights sum to 1
        mComboIW3Out = mComboIW3Out / mComboIW3Out.sum()
        
        # Clear for next models
        del hold, w3
        
        
        
        
        #################### INVERSE WEIGHT with weights to 10th power
        w4 = pow(w,10)
        w4 = w4.div(w4.sum(axis=1), axis=0)
        hold = np.dot(df_merged_red.fillna(0),w4.T)
        mComboIW4Out = pd.DataFrame(hold, index=df_merged_red.index, columns=['PDF_comboIW4'])

        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboIW4Out.at['PO','PDF_comboIW4'] = 0
            
        # Ensure final weights sum to 1
        mComboIW4Out = mComboIW4Out / mComboIW4Out.sum()
        
        # Clear for next models
        del hold, w4
        
        
        
        
        #################### INVERSE WEIGHT with weights to 20th power
        w5 = pow(w,20)
        w5 = w5.div(w5.sum(axis=1), axis=0)
        hold = np.dot(df_merged_red.fillna(0),w5.T)
        mComboIW5Out = pd.DataFrame(hold, index=df_merged_red.index, columns=['PDF_comboIW5'])

        # Set PO (pitchout) to zero if there are no runners on base
        if runner_on is False:
            mComboIW5Out.at['PO','PDF_comboIW5'] = 0
            
        # Ensure final weights sum to 1
        mComboIW5Out = mComboIW5Out / mComboIW5Out.sum()
        
        # Clear for next models
        del hold, w5
        
        
    else:
        mComboEWOut = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboEW'])
        mComboEW2Out = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboEW2'])
        mComboIWOut = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboIW'])
        mComboIW2Out = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboIW2'])
        mComboIW3Out = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboIW3'])
        mComboIW4Out = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboIW4'])
        mComboIW5Out = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_comboIW5'])



    # Output
    return mComboEWOut, mComboEW2Out, mComboIWOut, mComboIW2Out, mComboIW3Out, mComboIW4Out, mComboIW5Out


