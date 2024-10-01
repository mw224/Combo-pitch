# funs.py

############################################################################################################################
# Model 1: global average
def modelOne(df_train):
    
    # Third-party imports
    import pandas as pd

    # Calculate PDF
    ct = df_train['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
    ct.rename(columns={'proportion':'PDF_m1'}, inplace=True)
    
    # Output
    return ct






############################################################################################################################
# Model 2: equal-weight (1/n)
def modelTwo(unique_pitches):
    
    # Third-party imports
    import pandas as pd
    
    # Create 1/n PDF
    n = len(unique_pitches)
    ct = [1/n] * n
    ct = pd.DataFrame(ct, columns=['PDF_m2'])
    ct = ct.set_index(unique_pitches['pitch_type'])
    
    # Output
    return ct




############################################################################################################################
# Model 3: batter handedness
def modelThree(df_train,batter_handedness):
    
    # Third-party imports
    import pandas as pd
    
    # Filter to batter's handedness
    df_train_red = df_train[df_train['stand'].isin([batter_handedness])]
    
    # Calculate PDF
    ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
    ct.rename(columns={'proportion':'PDF_m3'}, inplace=True)
    
    # Output
    return ct



############################################################################################################################
# Model 4: Batter and pitcher handedness
def modelFour(df_train,batter_handedness,pitcher_handedness):
    
    # Third-party imports
    import pandas as pd
    
    # Filter to batter's handedness    
    df_train_red = df_train[(df_train['stand'].isin([batter_handedness])) & (df_train['p_throws'].isin([pitcher_handedness]))]
    
    # Calculate PDF
    ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
    ct.rename(columns={'proportion':'PDF_m4'}, inplace=True)
    
    # Output
    return ct
    




############################################################################################################################
# Model 5: No runners on base
def modelFive(df_train,unique_pitches,runner_on):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on == False:
        # Filter to base configuration
        df_train_red = df_train[df_train[['on_1b','on_2b','on_3b']].isna().all(1)]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m5'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m5'])
    
    # Output
    return ct





############################################################################################################################
# Model 6: Runner on 1st
def modelSix(df_train,unique_pitches,runner_on_1):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_1 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train['on_1b'].notnull()) & (df_train[['on_2b','on_3b']].isna().all(1))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m6'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m6'])
    
    # Output
    return ct






############################################################################################################################
# Model 7: Runner on 2nd
def modelSeven(df_train,unique_pitches,runner_on_2):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_2 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train['on_2b'].notnull()) & (df_train[['on_1b','on_3b']].isna().all(1))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m7'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m7'])
    
    # Output
    return ct






############################################################################################################################
# Model 8: Runner on 3rd
def modelEight(df_train,unique_pitches,runner_on_3):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_3 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train['on_3b'].notnull()) & (df_train[['on_1b','on_2b']].isna().all(1))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m8'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m8'])
    
    # Output
    return ct





############################################################################################################################
# Model 9: Runner on 1st and 2nd
def modelNine(df_train,unique_pitches,runner_on_12):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_12 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_1b','on_2b']].notnull().all(1)) & (df_train['on_3b'].isna())]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m9'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m9'])
    
    # Output
    return ct





############################################################################################################################
# Model 10: Runner on 1st and 3rd
def modelTen(df_train,unique_pitches,runner_on_13):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_13 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_1b','on_3b']].notnull().all(1)) & (df_train['on_2b'].isna())]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m10'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m10'])
    
    # Output
    return ct






############################################################################################################################
# Model 11: Runner on 2nd and 3rd
def modelEleven(df_train,unique_pitches,runner_on_23):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_23 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_2b','on_3b']].notnull().all(1)) & (df_train['on_1b'].isna())]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m11'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m11'])
    
    # Output
    return ct





############################################################################################################################
# Model 12: Runner on 1st, 2nd, and 3rd
def modelTwelve(df_train,unique_pitches,runner_on_123):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_123 == True:
        # Filter to base configuration
        df_train_red = df_train[df_train[['on_1b','on_2b','on_3b']].notnull().all(1)]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m12'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m12'])
    
    # Output
    return ct





############################################################################################################################
# Model 13: pitcher-specific average
def modelThirteen(df_train,pitcher_id):
    
    # Third-party imports
    import pandas as pd
    
    # Filter to pitcher's id
    df_train_red = df_train[df_train['pitcher'].isin([pitcher_id])]
    
    # Calculate PDF
    ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
    ct.rename(columns={'proportion':'PDF_m13'}, inplace=True)
    
    # Output
    return ct




############################################################################################################################
# Model 14: pitcher-batter history
def modelFourteen(df_train,pitcher_id,batter_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to pitcher's id
    df_train_red = df_train[(df_train['pitcher'].isin([pitcher_id])) & (df_train['batter'].isin([batter_id]))]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m14'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m14'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 15: this pitcher versus this handedness
def modelFifteen(df_train,pitcher_id,batter_handedness,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to pitcher's id
    df_train_red = df_train[(df_train['pitcher'].isin([pitcher_id])) & (df_train['stand'].isin([batter_handedness]))]

    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m15'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m15'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 16: pitches thrown at this park
def modelSixteen(df_train,park_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to park id
    df_train_red = df_train[df_train['home_team'].isin([park_id])]

    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m16'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m16'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 17: count: 0-0
def modelSeventeen(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 0-0 count
    if (balls_i == 0) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m17'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m17'])

    
    # Output
    return ct



############################################################################################################################
# Model 18: this pitcher on 0-0 count
def modelEighteen(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 0) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m18'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m18'])

    # Output
    return ct





############################################################################################################################
# Model 19: all pitchers with 0 outs
def modelNineteen(df_train,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 0):
        # Filter to this many outs
        df_train_red = df_train[df_train['outs_when_up'].isin([outs_i])]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m19'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m19'])

    # Output
    return ct



############################################################################################################################
# Model 20: this pitcher with 0 outs
def modelTwenty(df_train,pitcher_id,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 0):
        # Filter to this pitcher with 0 outs
        df_train_red = df_train[(df_train['outs_when_up'].isin([outs_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m20'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m20'])
    
    # Output
    return ct




############################################################################################################################
# Model 21: Top of 1st inning
def modelTwentyOne(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 1) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m21'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m21'])
    
    # Output
    return ct




############################################################################################################################
# Model 22: Bottom of 1st inning
def modelTwentyTwo(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 1) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m22'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m22'])
    
    # Output
    return ct






############################################################################################################################
# Model 23: Top of 2nd inning
def modelTwentyThree(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 2) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m23'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m23'])
    
    # Output
    return ct





############################################################################################################################
# Model 24: Bottom of 2nd inning
def modelTwentyFour(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 2) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m24'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m24'])
    
    # Output
    return ct




############################################################################################################################
# Model 25: Top of 3rd inning
def modelTwentyFive(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 3) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m25'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m25'])
    
    # Output
    return ct




############################################################################################################################
# Model 26: Bottom of 3rd inning
def modelTwentySix(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 3) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m26'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m26'])
    
    # Output
    return ct





############################################################################################################################
# Model 27: Top of 4th inning
def modelTwentySeven(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 4) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m27'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m27'])
    
    # Output
    return ct





############################################################################################################################
# Model 28: Bottom of 4th inning
def modelTwentyEight(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 4) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m28'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m28'])
    
    # Output
    return ct




############################################################################################################################
# Model 29: Top of 5th inning
def modelTwentyNine(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 5) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m29'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m29'])
    
    # Output
    return ct





############################################################################################################################
# Model 30: Bottom of 5th inning
def modelThirty(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 5) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m30'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m30'])
    
    # Output
    return ct





############################################################################################################################
# Model 31: Top of 6th inning
def modelThirtyOne(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 6) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m31'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m31'])
    
    # Output
    return ct





############################################################################################################################
# Model 32: Bottom of 6th inning
def modelThirtyTwo(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 6) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m32'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m32'])
    
    # Output
    return ct





############################################################################################################################
# Model 33: Top of 7th inning
def modelThirtyThree(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 7) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m33'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m33'])
    
    # Output
    return ct





############################################################################################################################
# Model 34: Bottom of 7th inning
def modelThirtyFour(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 7) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m34'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m34'])
    
    # Output
    return ct





############################################################################################################################
# Model 35: Top of 8th inning
def modelThirtyFive(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 8) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m35'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m35'])
    
    # Output
    return ct





############################################################################################################################
# Model 36: Bottom of 8th inning
def modelThirtySix(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 8) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m36'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m36'])
    
    # Output
    return ct





############################################################################################################################
# Model 37: Top of 9th inning
def modelThirtySeven(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 9) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m37'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m37'])
    
    # Output
    return ct





############################################################################################################################
# Model 38: Bottom of 9th inning
def modelThirtyEight(df_train,inning_i,inning_topbot_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 9) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m38'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m38'])
    
    # Output
    return ct




############################################################################################################################
# Model 39: 10+ inning
def modelThirtyNine(df_train,inning_i,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if inning_i > 9:
        # Filter to inning configuration
        df_train_red = df_train[df_train['inning'] > 9]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m39'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m39'])
    
    # Output
    return ct





############################################################################################################################
# Model 40: count: 0-1
def modelForty(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 0-1 count
    if (balls_i == 0) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m40'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m40'])

    
    # Output
    return ct





############################################################################################################################
# Model 41: count: 0-2
def modelFortyOne(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 0-2 count
    if (balls_i == 0) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m41'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m41'])

    
    # Output
    return ct




############################################################################################################################
# Model 42: count: 1-0
def modelFortyTwo(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 1-0 count
    if (balls_i == 1) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m42'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m42'])

    
    # Output
    return ct




############################################################################################################################
# Model 43: count: 1-1
def modelFortyThree(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 1-1 count
    if (balls_i == 1) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m43'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m43'])

    
    # Output
    return ct





############################################################################################################################
# Model 44: count: 1-2
def modelFortyFour(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 1-2 count
    if (balls_i == 1) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m44'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m44'])

    
    # Output
    return ct



############################################################################################################################
# Model 45: count: 2-0
def modelFortyFive(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 2-0 count
    if (balls_i == 2) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m45'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m45'])

    
    # Output
    return ct





############################################################################################################################
# Model 46: count: 2-1
def modelFortySix(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 2-1 count
    if (balls_i == 2) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m46'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m46'])

    
    # Output
    return ct





############################################################################################################################
# Model 47: count: 2-2
def modelFortySeven(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 2-2 count
    if (balls_i == 2) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m47'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m47'])

    
    # Output
    return ct





############################################################################################################################
# Model 48: count: 3-0
def modelFortyEight(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 3-0 count
    if (balls_i == 3) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m48'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m48'])

    
    # Output
    return ct





############################################################################################################################
# Model 49: count: 3-1
def modelFortyNine(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 3-1 count
    if (balls_i == 3) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m49'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m49'])

    
    # Output
    return ct





############################################################################################################################
# Model 50: count: 3-2
def modelFifty(df_train,balls_i,strikes_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to 3-2 count
    if (balls_i == 3) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m50'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m50'])

    
    # Output
    return ct





############################################################################################################################
# Model 51: pitches thrown to this particular batter
def modelFiftyOne(df_train,batter_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to batter's id
    df_train_red = df_train[df_train['batter'].isin([batter_id])]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m51'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m51'}, inplace=True)
    
    # Output
    return ct




############################################################################################################################
# Model 52: pitches thrown by handedness of pitcher by all pitchers
def modelFiftyTwo(df_train,pitcher_handedness,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to pitcher's handedness
    df_train_red = df_train[df_train['p_throws'].isin([pitcher_handedness])]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m52'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m52'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 53: pitches thrown by by all pitchers at this score spread
def modelFiftyThree(df_train,score_spread_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to this score spread
    ss = (df_train['home_score'] - df_train['away_score']).to_frame()
    ss.rename(columns={0:'score_spread'}, inplace=True)
    df_train = pd.concat([df_train,ss], join='outer', axis=1)
    df_train_red = df_train[df_train['score_spread'].isin([score_spread_i])]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m53'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m53'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 54: pitches thrown by by this pitcher at this score spread
def modelFiftyFour(df_train,score_spread_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to this score spread for this pitcher
    ss = (df_train['home_score'] - df_train['away_score']).to_frame()
    ss.rename(columns={0:'score_spread'}, inplace=True)
    df_train = pd.concat([df_train,ss], join='outer', axis=1)
    df_train_red = df_train[(df_train['score_spread'].isin([score_spread_i])) & (df_train['pitcher'].isin([pitcher_id]))]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m54'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m54'}, inplace=True)
    
    # Output
    return ct




############################################################################################################################
# Model 55: this pitcher with 1 outs
def modelFiftyFive(df_train,pitcher_id,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 1):
        # Filter to this pitcher with 0 outs
        df_train_red = df_train[(df_train['outs_when_up'].isin([outs_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m55'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m55'])
    
    # Output
    return ct





############################################################################################################################
# Model 56: this pitcher with 2 outs
def modelFiftySix(df_train,pitcher_id,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 2):
        # Filter to this pitcher with 0 outs
        df_train_red = df_train[(df_train['outs_when_up'].isin([outs_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m56'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m56'])
    
    # Output
    return ct


############################################################################################################################
# Model 57: all pitchers with 1 outs
def modelFiftySeven(df_train,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 1):
        # Filter to this many outs
        df_train_red = df_train[df_train['outs_when_up'].isin([outs_i])]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m57'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m57'])

    # Output
    return ct




############################################################################################################################
# Model 58: all pitchers with 2 outs
def modelFiftyEight(df_train,outs_i,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (outs_i == 2):
        # Filter to this many outs
        df_train_red = df_train[df_train['outs_when_up'].isin([outs_i])]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m58'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m58'])

    # Output
    return ct




############################################################################################################################
# Model 59: pitches thrown to this batter from the pitcher's team
def modelFiftyNine(df_train,pitching_team_i,unique_pitches,batter_id):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to pitcher's team and this batter
    df_train_red = df_train[((df_train['home_team'].isin([pitching_team_i])) | (df_train['away_team'].isin([pitching_team_i]))) & (df_train['batter'].isin([batter_id]))]
    
    # Calculate PDF
    if df_train_red.empty:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m59'])
    else:
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m59'}, inplace=True)
    
    # Output
    return ct





############################################################################################################################
# Model 60: this pitcher on 0-1 count
def modelSixty(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 0) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m60'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m60'])

    # Output
    return ct




############################################################################################################################
# Model 61: this pitcher on 0-2 count
def modelSixtyOne(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 0) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m61'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m61'])

    # Output
    return ct




############################################################################################################################
# Model 62: this pitcher on 1-0 count
def modelSixtyTwo(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 1) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m62'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m62'])

    # Output
    return ct




############################################################################################################################
# Model 63: this pitcher on 1-1 count
def modelSixtyThree(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 1) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m63'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m63'])

    # Output
    return ct




############################################################################################################################
# Model 64: this pitcher on 1-2 count
def modelSixtyFour(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 1) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m64'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m64'])

    # Output
    return ct





############################################################################################################################
# Model 65: this pitcher on 2-0 count
def modelSixtyFive(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 2) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m65'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m65'])

    # Output
    return ct




############################################################################################################################
# Model 66: this pitcher on 2-1 count
def modelSixtySix(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 2) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m66'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m66'])

    # Output
    return ct




############################################################################################################################
# Model 67: this pitcher on 2-2 count
def modelSixtySeven(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 2) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m67'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m67'])

    # Output
    return ct




############################################################################################################################
# Model 68: this pitcher on 3-0 count
def modelSixtyEight(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 3) and (strikes_i == 0):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m68'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m68'])

    # Output
    return ct




############################################################################################################################
# Model 69: this pitcher on 3-1 count
def modelSixtyNine(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 3) and (strikes_i == 1):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m69'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m69'])

    # Output
    return ct




############################################################################################################################
# Model 70: this pitcher on 3-2 count
def modelSeventy(df_train,balls_i,strikes_i,pitcher_id,unique_pitches):
    
    # Third-party imports
    import pandas as pd
    import numpy as np
    
    # Filter to count for this particular pitcher
    if (balls_i == 3) and (strikes_i == 2):
        df_train_red = df_train[(df_train['balls'].isin([balls_i])) & (df_train['strikes'].isin([strikes_i])) & (df_train['pitcher'].isin([pitcher_id]))]

        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m70'}, inplace=True)
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m70'])

    # Output
    return ct




############################################################################################################################
# Model 71: This pitcher with no runners on base
def modelSeventyOne(df_train,unique_pitches,runner_on,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on == False:
        # Filter to base configuration for this pitcher 
        df_train_red = df_train[(df_train[['on_1b','on_2b','on_3b']].isna().all(1)) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m71'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m71'])
    
    # Output
    return ct




############################################################################################################################
# Model 72: This pitcher with runner on 1st
def modelSeventyTwo(df_train,unique_pitches,runner_on_1,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_1 == True:
        # Filter to base configuration for this pitcher
        df_train_red = df_train[(df_train['on_1b'].notnull()) & (df_train[['on_2b','on_3b']].isna().all(1)) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m72'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m72'])
    
    # Output
    return ct




############################################################################################################################
# Model 73: This pitcher with runner on 2nd
def modelSeventyThree(df_train,unique_pitches,runner_on_2,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_2 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train['on_2b'].notnull()) & (df_train[['on_1b','on_3b']].isna().all(1)) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m73'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m73'])
    
    # Output
    return ct




############################################################################################################################
# Model 74: This pitcher with runner on 3rd
def modelSeventyFour(df_train,unique_pitches,runner_on_3,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_3 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train['on_3b'].notnull()) & (df_train[['on_1b','on_2b']].isna().all(1)) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m74'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m74'])
    
    # Output
    return ct




############################################################################################################################
# Model 75: This pitcher with runner on 1st and 2nd
def modelSeventyFive(df_train,unique_pitches,runner_on_12,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_12 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_1b','on_2b']].notnull().all(1)) & (df_train['on_3b'].isna()) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m75'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m75'])
    
    # Output
    return ct




############################################################################################################################
# Model 76: This pitcher with runner on 1st and 3rd
def modelSeventySix(df_train,unique_pitches,runner_on_13,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_13 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_1b','on_3b']].notnull().all(1)) & (df_train['on_2b'].isna()) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m76'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m76'])
    
    # Output
    return ct





############################################################################################################################
# Model 77: This pitcher with runner on 2nd and 3rd
def modelSeventySeven(df_train,unique_pitches,runner_on_23,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_23 == True:
        # Filter to base configuration        
        df_train_red = df_train[(df_train[['on_2b','on_3b']].notnull().all(1)) & (df_train['on_1b'].isna()) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m77'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m77'])
    
    # Output
    return ct




############################################################################################################################
# Model 78: This pitcher with runner on 1st, 2nd, and 3rd
def modelSeventyEight(df_train,unique_pitches,runner_on_123,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if runner_on_123 == True:
        # Filter to base configuration
        df_train_red = df_train[(df_train[['on_1b','on_2b','on_3b']].notnull().all(1)) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m78'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m78'])
    
    # Output
    return ct





############################################################################################################################
# Model 79: This pitcher in the top of 1st inning
def modelSeventyNine(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 1) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m79'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m79'])
    
    # Output
    return ct




############################################################################################################################
# Model 80: This pitcher in the bottom of 1st inning
def modelEighty(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 1) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m80'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m80'])
    
    # Output
    return ct






############################################################################################################################
# Model 81: This pitcher in the top of 2nd inning
def modelEightyOne(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 2) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m81'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m81'])
    
    # Output
    return ct





############################################################################################################################
# Model 82: This pitcher in the bottom of 2nd inning
def modelEightyTwo(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 2) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m82'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m82'])
    
    # Output
    return ct




############################################################################################################################
# Model 83: This pitcher in the top of 3rd inning
def modelEightyThree(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 3) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m83'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m83'])
    
    # Output
    return ct




############################################################################################################################
# Model 84: This pitcher in the bottom of 3rd inning
def modelEightyFour(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 3) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m84'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m84'])
    
    # Output
    return ct





############################################################################################################################
# Model 85: This pitcher in the top of 4th inning
def modelEightyFive(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 4) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m85'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m85'])
    
    # Output
    return ct





############################################################################################################################
# Model 86: This pitcher in the bottom of 4th inning
def modelEightySix(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 4) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m86'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m86'])
    
    # Output
    return ct




############################################################################################################################
# Model 87: This pitcher in the top of 5th inning
def modelEightySeven(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 5) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m87'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m87'])
    
    # Output
    return ct





############################################################################################################################
# Model 88: This pitcher in the bottom of 5th inning
def modelEightyEight(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 5) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m88'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m88'])
    
    # Output
    return ct





############################################################################################################################
# Model 89: This pitcher in the top of 6th inning
def modelEightyNine(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 6) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m89'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m89'])
    
    # Output
    return ct





############################################################################################################################
# Model 90: This pitcher in the bottom of 6th inning
def modelNinety(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 6) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m90'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m90'])
    
    # Output
    return ct





############################################################################################################################
# Model 91: This pitcher in the top of 7th inning
def modelNinetyOne(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 7) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m91'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m91'])
    
    # Output
    return ct





############################################################################################################################
# Model 92: This pitcher in the bottom of 7th inning
def modelNinetyTwo(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 7) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m92'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m92'])
    
    # Output
    return ct





############################################################################################################################
# Model 93: This pitcher in the top of 8th inning
def modelNinetyThree(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 8) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m93'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m93'])
    
    # Output
    return ct




############################################################################################################################
# Model 94: This pitcher in the Bottom of 8th inning
def modelNinetyFour(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 8) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m94'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m94'])
    
    # Output
    return ct





############################################################################################################################
# Model 95: This pitcher in the top of 9th inning
def modelNinetyFive(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 9) and (inning_topbot_i == "Top"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m95'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m95'])
    
    # Output
    return ct





############################################################################################################################
# Model 96: This pitcher in the bottom of 9th inning
def modelNinetySix(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if (inning_i == 9) and (inning_topbot_i == "Bot"):
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'].isin([inning_i])) & (df_train['inning_topbot'].isin([inning_topbot_i])) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m96'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m96'])
    
    # Output
    return ct




############################################################################################################################
# Model 97: This pitcher in the 10+ inning
def modelNinetySeven(df_train,inning_i,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if inning_i > 9:
        # Filter to inning configuration
        df_train_red = df_train[(df_train['inning'] > 9) & (df_train['pitcher'].isin([pitcher_id]))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m97'}, inplace=True)

    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m97'])
    
    # Output
    return ct




############################################################################################################################
# Model 98: Last pitch thrown
def modelNinetyEight(df_train,lp,unique_pitches):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lp != 'nan':
        df_train_flip = df_train.iloc[::-1]
        df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.pitch_type.shift().eq(lp))]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m98'}, inplace=True)
        
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m98'])
    
    # Output
    return ct




############################################################################################################################
# Model 99: Last pitch thrown for this particular pitcher
def modelNinetyNine(df_train,lp,unique_pitches,pitcher_id):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lp != 'nan':
        df_train_flip = df_train.iloc[::-1]
        df_train_red = df_train_flip[df_train_flip.batter.eq(df_train_flip.batter.shift()) & \
            df_train_flip.pitcher.eq(df_train_flip.pitcher.shift()) & df_train_flip.pitch_type.shift().eq(lp) & \
            df_train_flip['pitcher'].isin([pitcher_id])]
        
        # Calculate PDF
        ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
        ct.rename(columns={'proportion':'PDF_m99'}, inplace=True)
        
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m99'])
    
    # Output
    return ct




############################################################################################################################
# Model 100: Last zone thrown is in zone 1
def modelOneHundred(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 1:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m100'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m100'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m100'])
    
    # Output
    return ct




############################################################################################################################
# Model 101: Last zone thrown is in zone 2
def modelOneHundredOne(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 2:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m101'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m101'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m101'])
    
    # Output
    return ct




############################################################################################################################
# Model 102: Last zone thrown is in zone 3
def modelOneHundredTwo(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 3:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m102'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m102'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m102'])
    
    # Output
    return ct




############################################################################################################################
# Model 103: Last zone thrown is in zone 4
def modelOneHundredThree(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 4:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m103'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m103'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m103'])
    
    # Output
    return ct




############################################################################################################################
# Model 104: Last zone thrown is in zone 5
def modelOneHundredFour(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 5:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m104'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m104'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m104'])
    
    # Output
    return ct




############################################################################################################################
# Model 105: Last zone thrown is in zone 6
def modelOneHundredFive(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 6:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m105'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m105'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m105'])
    
    # Output
    return ct




############################################################################################################################
# Model 106: Last zone thrown is in zone 7
def modelOneHundredSix(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 7:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m106'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m106'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m106'])
    
    # Output
    return ct




############################################################################################################################
# Model 107: Last zone thrown is in zone 8
def modelOneHundredSeven(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 8:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m107'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m107'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m107'])
    
    # Output
    return ct




############################################################################################################################
# Model 108: Last zone thrown is in zone 9
def modelOneHundredEight(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 9:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m108'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m108'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m108'])
    
    # Output
    return ct



############################################################################################################################
# Model 109: Last zone thrown is in zone 11
def modelOneHundredNine(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 11:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m109'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m109'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m109'])
    
    # Output
    return ct




############################################################################################################################
# Model 110: Last zone thrown is in zone 12
def modelOneHundredTen(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 12:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m110'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m110'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m110'])
    
    # Output
    return ct





############################################################################################################################
# Model 111: Last zone thrown is in zone 13
def modelOneHundredEleven(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 13:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m111'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m111'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m111'])
    
    # Output
    return ct




############################################################################################################################
# Model 112: Last zone thrown is in zone 14
def modelOneHundredTwelve(df_train,unique_pitches,lpz):

    # Third-party imports
    import pandas as pd
    import numpy as np
    
    if lpz != 'nan':
        if lpz == 14:
            df_train_flip = df_train.iloc[::-1]
            df_train_red = df_train_flip[(df_train_flip.batter.eq(df_train_flip.batter.shift())) & (df_train_flip.pitcher.eq(df_train_flip.pitcher.shift())) & (df_train_flip.zone.shift().eq(lpz))]
            
            # Calculate PDF
            ct = df_train_red['pitch_type'].value_counts(normalize=True, dropna=True).to_frame()
            ct.rename(columns={'proportion':'PDF_m112'}, inplace=True)
        else:
            ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m112'])
    else:
        ct = pd.DataFrame(np.nan, index=unique_pitches['pitch_type'], columns=['PDF_m112'])
    
    # Output
    return ct






