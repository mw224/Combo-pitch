# pitchSelection.py

# Third-party imports
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import timedelta, datetime
from functools import reduce

# First-party imports
from trainAndTest import trainAndTest
from comboModel import comboModel
from runnerOn import runnerOn
from dataBrierScore import residualBrierScore
from funs import modelOne, modelTwo, modelThree, modelFour, modelFive, modelSix, \
    modelSeven, modelEight, modelNine, modelTen, modelEleven, modelTwelve, \
    modelThirteen, modelFourteen, modelFifteen, modelSixteen, modelSeventeen, \
    modelEighteen, modelNineteen, modelTwenty, modelTwentyOne, modelTwentyTwo, \
    modelTwentyThree, modelTwentyFour, modelTwentyFive, modelTwentySix, modelTwentySeven, \
    modelTwentyEight, modelTwentyNine, modelThirty, modelThirtyOne, modelThirtyTwo, \
    modelThirtyThree, modelThirtyFour, modelThirtyFive, modelThirtySix, modelThirtySeven, \
    modelThirtyEight, modelThirtyNine, modelForty, modelFortyOne, modelFortyTwo, \
    modelFortyThree, modelFortyFour, modelFortyFive, modelFortySix, modelFortySeven, \
    modelFortyEight, modelFortyNine, modelFifty, modelFiftyOne, modelFiftyTwo, \
    modelFiftyThree, modelFiftyFour, modelFiftyFive, modelFiftySix, modelFiftySeven, \
    modelFiftyEight, modelFiftyNine, modelSixty, modelSixtyOne, modelSixtyTwo, \
    modelSixtyThree, modelSixtyFour, modelSixtyFive, modelSixtySix, modelSixtySeven, \
    modelSixtyEight, modelSixtyNine, modelSeventy, modelSeventyOne, modelSeventyTwo, \
    modelSeventyThree, modelSeventyFour, modelSeventyFive, modelSeventySix, modelSeventySeven, \
    modelSeventyEight, modelSeventyNine, modelEighty, modelEightyOne, modelEightyTwo, \
    modelEightyThree, modelEightyFour, modelEightyFive, modelEightySix, modelEightySeven, \
    modelEightyEight, modelEightyNine, modelNinety, modelNinetyOne, modelNinetyTwo, \
    modelNinetyThree, modelNinetyFour, modelNinetyFive, modelNinetySix, modelNinetySeven, \
    modelNinetyEight, modelNinetyNine, modelOneHundred, modelOneHundredOne, \
    modelOneHundredTwo, modelOneHundredThree, modelOneHundredFour, modelOneHundredFive, \
    modelOneHundredSix, modelOneHundredSeven, modelOneHundredEight, modelOneHundredNine, \
    modelOneHundredTen, modelOneHundredEleven, modelOneHundredTwelve
    


#### MANUAL DATE ENTRIES
start_date = "2023-07-14"
oos_date = "2023-07-24"


# Create training and testing data sets
df_train, df_test, unique_pitches = trainAndTest(start_date)

# Loop through each unique game, and then within that work through each pitch
ng = pd.DataFrame(df_test.game_pk.unique())
ng.rename(columns={0: 'game_pk'}, inplace=True)
R = pd.DataFrame(np.empty(0))
R_oos = R
BS = R
for i in range(len(ng)):
    # Filter to the particular game
    gpi = ng['game_pk'].iloc[i]
    df_test_i = df_test[df_test['game_pk'] == gpi]
    df_test_i = df_test_i.iloc[::-1]
    game_date_i = df_test_i['game_date'].iloc[0]
    
    start_time = time.perf_counter()
    
    # Loop through each pitch in the particular game
    for j in range(len(df_test_i)):
        
        # Pitch to forecast (i.e., realized value)
        p = df_test_i['pitch_type'].iloc[j]
        z = df_test_i['zone'].iloc[j]
        
        # Independent variables
        batter_handedness = df_test_i['stand'].iloc[j]
        pitcher_handedness = df_test_i['p_throws'].iloc[j]
        runner_on, runner_on_1, runner_on_2, runner_on_3, runner_on_12, \
            runner_on_13, runner_on_23, runner_on_123 = runnerOn(df_test_i.iloc[j].to_frame())
        pitcher_id = df_test_i['pitcher'].iloc[j]
        batter_id = df_test_i['batter'].iloc[j]
        park_id = df_test_i['home_team'].iloc[j]
        balls_i = df_test_i['balls'].iloc[j]
        strikes_i = df_test_i['strikes'].iloc[j]
        outs_i = df_test_i['outs_when_up'].iloc[j]
        inning_i = df_test_i['inning'].iloc[j]
        inning_topbot_i = df_test_i['inning_topbot'].iloc[j]
        score_spread_i = df_test_i['home_score'].iloc[j] - df_test_i['away_score'].iloc[j]
        if inning_topbot_i == "Top":
            pitching_team_i = df_test_i['home_team'].iloc[j]
        else:
            pitching_team_i = df_test_i['away_team'].iloc[j]   
        if j > 0:
            lp = lp_j
            lpz = lpz_j
        else:
            lp = 'nan'
            lpz = 'nan'
        
        # Pitch selection forecasting models
        m1Out = modelOne(df_train)
        m2Out = modelTwo(unique_pitches)
        m3Out = modelThree(df_train,batter_handedness)
        m4Out = modelFour(df_train,batter_handedness,pitcher_handedness)
        m5Out = modelFive(df_train,unique_pitches,runner_on)
        m6Out = modelSix(df_train,unique_pitches,runner_on_1)
        m7Out = modelSeven(df_train,unique_pitches,runner_on_2)
        m8Out = modelEight(df_train,unique_pitches,runner_on_3)
        m9Out = modelNine(df_train,unique_pitches,runner_on_12)
        m10Out = modelTen(df_train,unique_pitches,runner_on_13)
        m11Out = modelEleven(df_train,unique_pitches,runner_on_23)
        m12Out = modelTwelve(df_train,unique_pitches,runner_on_123)
        m13Out = modelThirteen(df_train,pitcher_id)
        m14Out = modelFourteen(df_train,pitcher_id,batter_id,unique_pitches)
        m15Out = modelFifteen(df_train,pitcher_id,batter_handedness,unique_pitches)
        m16Out = modelSixteen(df_train,park_id,unique_pitches)
        m17Out = modelSeventeen(df_train,balls_i,strikes_i,unique_pitches)
        m18Out = modelEighteen(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m19Out = modelNineteen(df_train,outs_i,unique_pitches)
        m20Out = modelTwenty(df_train,pitcher_id,outs_i,unique_pitches)
        m21Out = modelTwentyOne(df_train,inning_i,inning_topbot_i,unique_pitches)
        m22Out = modelTwentyTwo(df_train,inning_i,inning_topbot_i,unique_pitches)
        m23Out = modelTwentyThree(df_train,inning_i,inning_topbot_i,unique_pitches)
        m24Out = modelTwentyFour(df_train,inning_i,inning_topbot_i,unique_pitches)
        m25Out = modelTwentyFive(df_train,inning_i,inning_topbot_i,unique_pitches)
        m26Out = modelTwentySix(df_train,inning_i,inning_topbot_i,unique_pitches)
        m27Out = modelTwentySeven(df_train,inning_i,inning_topbot_i,unique_pitches)
        m28Out = modelTwentyEight(df_train,inning_i,inning_topbot_i,unique_pitches)
        m29Out = modelTwentyNine(df_train,inning_i,inning_topbot_i,unique_pitches)
        m30Out = modelThirty(df_train,inning_i,inning_topbot_i,unique_pitches)
        m31Out = modelThirtyOne(df_train,inning_i,inning_topbot_i,unique_pitches)
        m32Out = modelThirtyTwo(df_train,inning_i,inning_topbot_i,unique_pitches)
        m33Out = modelThirtyThree(df_train,inning_i,inning_topbot_i,unique_pitches)
        m34Out = modelThirtyFour(df_train,inning_i,inning_topbot_i,unique_pitches)
        m35Out = modelThirtyFive(df_train,inning_i,inning_topbot_i,unique_pitches)
        m36Out = modelThirtySix(df_train,inning_i,inning_topbot_i,unique_pitches)
        m37Out = modelThirtySeven(df_train,inning_i,inning_topbot_i,unique_pitches)
        m38Out = modelThirtyEight(df_train,inning_i,inning_topbot_i,unique_pitches)
        m39Out = modelThirtyNine(df_train,inning_i,unique_pitches)
        m40Out = modelForty(df_train,balls_i,strikes_i,unique_pitches)
        m41Out = modelFortyOne(df_train,balls_i,strikes_i,unique_pitches)
        m42Out = modelFortyTwo(df_train,balls_i,strikes_i,unique_pitches)
        m43Out = modelFortyThree(df_train,balls_i,strikes_i,unique_pitches)
        m44Out = modelFortyFour(df_train,balls_i,strikes_i,unique_pitches)
        m45Out = modelFortyFive(df_train,balls_i,strikes_i,unique_pitches)
        m46Out = modelFortySix(df_train,balls_i,strikes_i,unique_pitches)
        m47Out = modelFortySeven(df_train,balls_i,strikes_i,unique_pitches)
        m48Out = modelFortyEight(df_train,balls_i,strikes_i,unique_pitches)
        m49Out = modelFortyNine(df_train,balls_i,strikes_i,unique_pitches)
        m50Out = modelFifty(df_train,balls_i,strikes_i,unique_pitches)
        m51Out = modelFiftyOne(df_train,batter_id,unique_pitches)
        m52Out = modelFiftyTwo(df_train,pitcher_handedness,unique_pitches)
        m53Out = modelFiftyThree(df_train,score_spread_i,unique_pitches)
        m54Out = modelFiftyFour(df_train,score_spread_i,pitcher_id,unique_pitches)
        m55Out = modelFiftyFive(df_train,pitcher_id,outs_i,unique_pitches)
        m56Out = modelFiftySix(df_train,pitcher_id,outs_i,unique_pitches)
        m57Out = modelFiftySeven(df_train,outs_i,unique_pitches)
        m58Out = modelFiftyEight(df_train,outs_i,unique_pitches)
        m59Out = modelFiftyNine(df_train,pitching_team_i,unique_pitches,batter_id)
        m60Out = modelSixty(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m61Out = modelSixtyOne(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m62Out = modelSixtyTwo(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m63Out = modelSixtyThree(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m64Out = modelSixtyFour(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m65Out = modelSixtyFive(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m66Out = modelSixtySix(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m67Out = modelSixtySeven(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m68Out = modelSixtyEight(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m69Out = modelSixtyNine(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m70Out = modelSeventy(df_train,balls_i,strikes_i,pitcher_id,unique_pitches)
        m71Out = modelSeventyOne(df_train,unique_pitches,runner_on,pitcher_id)
        m72Out = modelSeventyTwo(df_train,unique_pitches,runner_on_1,pitcher_id)
        m73Out = modelSeventyThree(df_train,unique_pitches,runner_on_2,pitcher_id)
        m74Out = modelSeventyFour(df_train,unique_pitches,runner_on_3,pitcher_id)
        m75Out = modelSeventyFive(df_train,unique_pitches,runner_on_12,pitcher_id)
        m76Out = modelSeventySix(df_train,unique_pitches,runner_on_13,pitcher_id)
        m77Out = modelSeventySeven(df_train,unique_pitches,runner_on_23,pitcher_id)
        m78Out = modelSeventyEight(df_train,unique_pitches,runner_on_123,pitcher_id)
        m79Out = modelSeventyNine(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m80Out = modelEighty(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m81Out = modelEightyOne(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m82Out = modelEightyTwo(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m83Out = modelEightyThree(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m84Out = modelEightyFour(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m85Out = modelEightyFive(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m86Out = modelEightySix(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m87Out = modelEightySeven(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m88Out = modelEightyEight(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m89Out = modelEightyNine(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m90Out = modelNinety(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m91Out = modelNinetyOne(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m92Out = modelNinetyTwo(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m93Out = modelNinetyThree(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m94Out = modelNinetyFour(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m95Out = modelNinetyFive(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m96Out = modelNinetySix(df_train,inning_i,inning_topbot_i,unique_pitches,pitcher_id)
        m97Out = modelNinetySeven(df_train,inning_i,unique_pitches,pitcher_id)
        m98Out = modelNinetyEight(df_train,lp,unique_pitches)
        m99Out = modelNinetyNine(df_train,lp,unique_pitches,pitcher_id)
        m100Out = modelOneHundred(df_train,unique_pitches,lpz)
        m101Out = modelOneHundredOne(df_train,unique_pitches,lpz)
        m102Out = modelOneHundredTwo(df_train,unique_pitches,lpz)
        m103Out = modelOneHundredThree(df_train,unique_pitches,lpz)
        m104Out = modelOneHundredFour(df_train,unique_pitches,lpz)
        m105Out = modelOneHundredFive(df_train,unique_pitches,lpz)
        m106Out = modelOneHundredSix(df_train,unique_pitches,lpz)
        m107Out = modelOneHundredSeven(df_train,unique_pitches,lpz)
        m108Out = modelOneHundredEight(df_train,unique_pitches,lpz)
        m109Out = modelOneHundredNine(df_train,unique_pitches,lpz)
        m110Out = modelOneHundredTen(df_train,unique_pitches,lpz)
        m111Out = modelOneHundredEleven(df_train,unique_pitches,lpz)
        m112Out = modelOneHundredTwelve(df_train,unique_pitches,lpz)        
                
        # Collect data for Brier scores
        dfs = [m1Out['PDF_m1'],m2Out['PDF_m2'],m3Out['PDF_m3'],m4Out['PDF_m4'], \
               m5Out['PDF_m5'],m6Out['PDF_m6'],m7Out['PDF_m7'],m8Out['PDF_m8'], \
               m9Out['PDF_m9'],m10Out['PDF_m10'],m11Out['PDF_m11'],m12Out['PDF_m12'], \
               m13Out['PDF_m13'],m14Out['PDF_m14'],m15Out['PDF_m15'],m16Out['PDF_m16'], \
               m17Out['PDF_m17'],m18Out['PDF_m18'],m19Out['PDF_m19'],m20Out['PDF_m20'], \
               m21Out['PDF_m21'],m22Out['PDF_m22'],m23Out['PDF_m23'],m24Out['PDF_m24'], \
               m25Out['PDF_m25'],m26Out['PDF_m26'],m27Out['PDF_m27'],m28Out['PDF_m28'], \
               m29Out['PDF_m29'],m30Out['PDF_m30'],m31Out['PDF_m31'],m32Out['PDF_m32'], \
               m33Out['PDF_m33'],m34Out['PDF_m34'],m35Out['PDF_m35'],m36Out['PDF_m36'], \
               m37Out['PDF_m37'],m38Out['PDF_m38'],m39Out['PDF_m39'],m40Out['PDF_m40'], \
               m41Out['PDF_m41'],m42Out['PDF_m42'],m43Out['PDF_m43'],m44Out['PDF_m44'], \
               m45Out['PDF_m45'],m46Out['PDF_m46'],m47Out['PDF_m47'],m48Out['PDF_m48'], \
               m49Out['PDF_m49'],m50Out['PDF_m50'],m51Out['PDF_m51'],m52Out['PDF_m52'], \
               m53Out['PDF_m53'],m54Out['PDF_m54'],m55Out['PDF_m55'],m56Out['PDF_m56'], \
               m57Out['PDF_m57'],m58Out['PDF_m58'],m59Out['PDF_m59'],m60Out['PDF_m60'], \
               m61Out['PDF_m61'],m62Out['PDF_m62'],m63Out['PDF_m63'],m64Out['PDF_m64'], \
               m65Out['PDF_m65'],m66Out['PDF_m66'],m67Out['PDF_m67'],m68Out['PDF_m68'], \
               m69Out['PDF_m69'],m70Out['PDF_m70'],m71Out['PDF_m71'],m72Out['PDF_m72'], \
               m73Out['PDF_m73'],m74Out['PDF_m74'],m75Out['PDF_m75'],m76Out['PDF_m76'], \
               m77Out['PDF_m77'],m78Out['PDF_m78'],m79Out['PDF_m79'],m80Out['PDF_m80'], \
               m81Out['PDF_m81'],m82Out['PDF_m82'],m83Out['PDF_m83'],m84Out['PDF_m84'], \
               m85Out['PDF_m85'],m86Out['PDF_m86'],m87Out['PDF_m87'],m88Out['PDF_m88'], \
               m89Out['PDF_m89'],m90Out['PDF_m90'],m91Out['PDF_m91'],m92Out['PDF_m92'], \
               m93Out['PDF_m93'],m94Out['PDF_m94'],m95Out['PDF_m95'],m96Out['PDF_m96'], \
               m97Out['PDF_m97'],m98Out['PDF_m98'],m99Out['PDF_m99'],m100Out['PDF_m100'], \
               m101Out['PDF_m101'],m102Out['PDF_m102'],m103Out['PDF_m103'],m104Out['PDF_m104'], \
               m105Out['PDF_m105'],m106Out['PDF_m106'],m107Out['PDF_m107'],m108Out['PDF_m108'], \
               m109Out['PDF_m109'],m110Out['PDF_m110'],m111Out['PDF_m111'],m112Out['PDF_m112']]
        df_merged = pd.concat(dfs, join='outer', axis=1)

        # Combine
        mComboEWOut, mComboEW2Out, mComboIWOut, mComboIW2Out, mComboIW3Out, mComboIW4Out,\
            mComboIW5Out = comboModel(df_merged,BS,unique_pitches,game_date_i,oos_date,runner_on)
        dfs_combo = [mComboEWOut['PDF_comboEW'], mComboEW2Out['PDF_comboEW2'], mComboIWOut['PDF_comboIW'], \
                mComboIW2Out['PDF_comboIW2'], mComboIW3Out['PDF_comboIW3'], mComboIW4Out['PDF_comboIW4'], \
                mComboIW5Out['PDF_comboIW5']]
        df_merged_combo = pd.concat(dfs_combo, join='outer', axis=1)
        df_merged_fin = pd.concat([df_merged,df_merged_combo], join='outer', axis = 1)
        
        # Update inputs for Brier score calculation
        resi = residualBrierScore(p,df_merged)
        R = pd.concat([R,resi], join='outer', axis=1)
        if game_date_i >= oos_date:
            resi_oos = residualBrierScore(p,df_merged_fin)
            R_oos = pd.concat([R_oos,resi_oos], join='outer', axis=1)

        # Save this pitch for models 98 and 99
        lp_j = p
        lpz_j = z

        # Clear for next pass
        del balls_i, batter_handedness, batter_id, df_merged, df_merged_fin, dfs, \
            inning_i, inning_topbot_i, lp, lpz, m1Out, m2Out, m3Out, m4Out, m5Out, \
            m6Out, m7Out, m8Out, m9Out, m10Out, m11Out, m12Out, m13Out, m14Out, \
            m15Out, m16Out, m17Out, m18Out, m19Out, m20Out, m21Out, m22Out, m23Out, \
            m24Out, m25Out, m26Out, m27Out, m28Out, m29Out, m30Out, m31Out, m32Out, \
            m33Out, m34Out, m35Out, m36Out, m37Out, m38Out, m39Out, m40Out, m41Out, \
            m42Out, m43Out, m44Out, m45Out, m46Out, m47Out, m48Out, m49Out, \
            m50Out, m51Out, m52Out, m53Out, m54Out, m55Out, m56Out, m57Out, m58Out, \
            m59Out, m60Out, m61Out, m62Out, m63Out, m64Out, m65Out, m66Out, m67Out, \
            m68Out, m69Out, m70Out, m71Out, m72Out, m73Out, m74Out, m75Out, m76Out, \
            m77Out, m78Out, m79Out, m80Out, m81Out, m82Out, m83Out, m84Out, m85Out, \
            m86Out, m87Out, m88Out, m89Out, m90Out, m91Out, m92Out, m93Out, m94Out, \
            m95Out, m96Out, m97Out, m98Out, m99Out, m100Out, m101Out, m102Out, m103Out, \
            m104Out, m105Out, m106Out, m107Out, m108Out, m109Out, m110Out, m111Out, \
            m112Out, outs_i, p, park_id, pitcher_handedness, pitcher_id, \
            pitching_team_i, resi, runner_on, runner_on_1, runner_on_2, runner_on_3, \
            runner_on_12, runner_on_13, runner_on_23, runner_on_123, score_spread_i, \
            strikes_i, z, df_merged_combo, dfs_combo, mComboEWOut, mComboEW2Out, \
            mComboIWOut, mComboIW2Out, mComboIW3Out, mComboIW4Out, mComboIW5Out

    # Print time to complete this game
    time_duration = timedelta(seconds=time.perf_counter() - start_time)
    print(time_duration)

    # Brier score for multiple categories for each model
    BS = R.mean(axis=1)
    
    # Output R and R_oos for backup
    temp_date = datetime.strptime(game_date_i, "%Y-%m-%d %H:%M:%S")
    date_str = temp_date.date()
    R1 = R.T
    R1.to_csv('R1out' + str(date_str) + '_' + str(gpi) + '.csv')
    R2 = R_oos.T
    R2.to_csv('R2out' + str(date_str) + '_' + str(gpi) + '.csv')
    
    # Clear for next pass
    del df_test_i, game_date_i, gpi
    
# This is the final Brier score using only the out-of-sample period for all models
BS_oos = R_oos.mean(axis=1)
    