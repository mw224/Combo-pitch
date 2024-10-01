# runnerOn.py

def runnerOn(df_test_i_i):
    
    # Third-party imports
    import pandas as pd
    import math
    
    # Any runners on base
    ron1 = df_test_i_i.at['on_1b',df_test_i_i.columns.array[0]]
    ron2 = df_test_i_i.at['on_2b',df_test_i_i.columns.array[0]]
    ron3 = df_test_i_i.at['on_3b',df_test_i_i.columns.array[0]]
    
    # Check now for individual base configurations
    if (math.isnan(ron1) == False) and (math.isnan(ron2) == False) and (math.isnan(ron3) == False):
        runner_on_123 = True
        runner_on_12 = False
        runner_on_13 = False
        runner_on_23 = False
        runner_on_1 = False
        runner_on_2 = False
        runner_on_3 = False
    else:
        runner_on_123 = False

        if (math.isnan(ron1) == False) and (math.isnan(ron2) == False):
            runner_on_12 = True
            runner_on_13 = False
            runner_on_23 = False
            runner_on_1 = False
            runner_on_2 = False
            runner_on_3 = False
        else:
            runner_on_12 = False
            
            if (math.isnan(ron1) == False) and (math.isnan(ron3) == False):
                runner_on_13 = True
                runner_on_23 = False
                runner_on_1 = False
                runner_on_2 = False
                runner_on_3 = False
            else:
                runner_on_13 = False
            
                if (math.isnan(ron2) == False) and (math.isnan(ron3) == False):
                    runner_on_23 = True
                    runner_on_1 = False
                    runner_on_2 = False
                    runner_on_3 = False
                else:
                    runner_on_23 = False
                    
                    if math.isnan(ron1):
                        runner_on_1 = False
                    else:
                        runner_on_1 = True
                    if math.isnan(ron2):
                        runner_on_2 = False
                    else:
                        runner_on_2 = True
                    if math.isnan(ron3):
                        runner_on_3 = False
                    else:
                        runner_on_3 = True
        
    if (runner_on_1 == True) or (runner_on_2 == True) or (runner_on_3 == True) or (runner_on_12 == True) \
        or (runner_on_13 == True) or (runner_on_23 == True) or (runner_on_123 == True):
            runner_on = True
    else:
        runner_on = False

    # Output
    return runner_on, runner_on_1, runner_on_2, runner_on_3, runner_on_12, runner_on_13, runner_on_23, runner_on_123
    
    
    
