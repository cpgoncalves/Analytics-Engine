# -*- coding: utf-8 -*-
"""
Carlos Pedro Goncalves, March, 2019
University of Lisbon
Instituto Superior de Ciências Sociais e Politicas (ISCSP)

Analytics Engine: 
    Analytics Engine is an AI-based system with data analysis automation built
    that works with pandas, stats, statsmodels and scikit-learn, it was 
    developed to work as a Business Analytics for Business Intelligence module,
    as part of classroom material, in the context of Statistics of Management 
    classes.
    
Copyright (c) 2019 Carlos Pedro Gonçalves

This work is licensed under the BSD 3-Clause License

@author: cpdsg
"""

import pandas as pd # pandas module
from scipy import stats # stats module
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
import numpy as np # numpy module
from matplotlib import pyplot as plt # pyplot module

# scikit-learn elements needed
from sklearn.metrics import accuracy_score # accuracy score
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import mean_squared_error # mean square error metric
from sklearn.metrics import mean_absolute_error # mean absolute error metric
from sklearn.metrics import median_absolute_error # median absolute error metric


def import_data(filename):  
    # convert to dataframe
    return pd.read_csv(filename)


def evaluate_normality_assumption(X,
                                  target_name,
                                  nan='propagate'):
    
    # evaluate the normality assumption for parametric tests
    
    print("\nTESTING NORMALITY ASSUMPTION FOR "+target_name)
    
    # apply normality test
    k2, p_value_normal = stats.normaltest(X,nan_policy=nan)
    
    # evaluate the normality assumption (using 1% significance to bias towards
    # the parametric tests)
    if p_value_normal > 0.01:
        assumption = True
        print("\nNormality assumption holds at 1% significance")
        print("Normal test significance", round(p_value_normal,6))
    else:
        assumption = False
        print("\nNormality assumption does not hold at 1% significance")
        print("Normal test significance", round(p_value_normal,6))
    # return the assumption output for the procedures involving the parametric
    # tests
    return assumption           
 
    
def statistics_quantitative(X,target_name):
    
    # printout main sample statistics along with skewness and kurtosis test
    print("\nMAIN SAMPLE STATISTICS FOR", target_name)
    print("\nSample mean", round(np.mean(X),6))
    print("Q1", round(np.percentile(X,q=25),6))
    print("Q2 (Median)", round(np.percentile(X,q=50),6))
    print("Q3", round(np.percentile(X,q=75)),6)
    print("Sample standard deviation", round(np.std(X,ddof=1),6))
    print("Minimum",np.min(X))
    print("Maximum",np.max(X))
    print("Skewness",round(stats.skew(X)),6)
    z_score, p_value_skewness = stats.skewtest(X)
    print("Skewness p-value", round(p_value_skewness,6))
    print("Kurtosis", round(stats.kurtosis(X),6))
    stat_test, p_value_kurtosis = stats.kurtosistest(X)
    print("Kurtosis p-value", round(p_value_kurtosis,6))
    
    
def one_mean(df, # pandas dataframe
             target_name, # target variable name
             popmean, # population mean
             alpha=0.05, # significance level
             nan='propagate' # nan policy
             ):
    
    # copy the data for the target variable
    X=df[target_name].values
    
    # printout the main sample statistics
    statistics_quantitative(X,target_name)
    
    # evaluate the normality assumption
    assumption = evaluate_normality_assumption(X,
                                               target_name,
                                               nan)
        
    # apply parametric test if normality assumption holds
    if assumption == True:
        # get the inferential statistics
        print("\nINFERENTIAL STATISTICS")
        mean, var, std = stats.mvsdist(X)
        # printout the confidence interval for the mean and 
        # the standard deviation
        print("\nBayesian", str((1-alpha)*100)+"%" , 
              "confidence interval for the mean:",
              mean.interval(1-alpha))
        print("Bayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation:",
              std.interval(1-alpha))
        
        print("\nONE SAMPLE TEST OF MEAN")
        # get the test statistics and p-value
        t, p_value = stats.mstats.ttest_onesamp(X, popmean, axis=0)
        
        # printout the test's main results
        print("\nTest statistic", round(t,6))
        print("p-value", round(p_value,6))
        if p_value > alpha:
            print("\nDo not reject the null hypothesis")
            print("Mean of", target_name,
                  "is not statistically different from",
                  popmean)
        else:
            print("\nReject the null hypothesis")
            print("Mean of", target_name, 
                  "is statistically different from", popmean)
    
    # output the boxplot for the target variable
    plt.boxplot(X,labels=[target_name])
    plt.show()


def two_means_indep(df, # pandas dataframe
                    target_name, # name of target variable
                    binary_variable, # binary variable name
                    binary_categories, # binary variable categories
                    alpha = 0.05, # significance level
                    nan = 'propagate' # nan policy
                    ):
            
    # copy the data for the two samples  
    
    name_0 = binary_categories[0] # name of first category
    name_1 = binary_categories[1] # name of second category
    
    df_0 = df.loc[df[binary_variable]==name_0] # sample for first category
    df_1 = df.loc[df[binary_variable]==name_1] # sample for second category
    
    X_0 = df_0[target_name].values # numpy array for first category
    X_1 = df_1[target_name].values # numpy array for second category
    
    # get the main sample statistics for both categories
    statistics_quantitative(X_0,name_0)
    statistics_quantitative(X_1,name_1)
    
    # evaluate normality assumption for both categories
    assumption_0 = evaluate_normality_assumption(X_0,
                                                 name_0,
                                                 nan)
    assumption_1 = evaluate_normality_assumption(X_1,
                                                 name_1,
                                                 nan)
        
    # apply test if normality assumption holds for both categories
    if assumption_0 == True and assumption_1 == True:
        # get the inferential statistics
        print("\nINFERENTIAL STATISTICS")
        mean_0, var_0, std_0 = stats.mvsdist(X_0)
        mean_1, var_1, std_1 = stats.mvsdist(X_1)
        
        print("\nBayesian", str((1-alpha)*100)+"%",
              "confidence interval for the mean of", name_0,":",
              mean_0.interval(1-alpha))
        print("\nBayesian", str((1-alpha)*100)+"%",
              "confidence interval for the mean of", name_1,":",
              mean_1.interval(1-alpha))
        
        print("\nBayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation of",
              name_0, ":",
              std_0.interval(1-alpha))
        print("\nBayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation of",
              name_1, ":",
              std_1.interval(1-alpha))
        
        # apply the Levene and the independent samples t-test
        print("\nLEVENE TEST FOR VARIANCES")
        W, p_value_levene = stats.levene(X_0,X_1)
        if p_value_levene > alpha:
            print("\nEqual variances assumed by Levene test")
            print("Significance of Levene test", round(p_value_levene,6))
            t, p_value = stats.ttest_ind(X_0,X_1,
                                         equal_var=True,
                                         nan_policy=nan)
        else:
            print("\nEqual variances not assumed by Levene test")
            print("Significance of Levene test", round(p_value_levene,6))
            t, p_value = stats.ttest_ind(X_0,X_1,
                                         equal_var=False,
                                         nan_policy=nan)
        
        # output the t-test's results
        print("\nTWO INDEPENDENT SAMPLES TEST OF MEAN")
        print("\nTest statistic", round(t,6))
        print("p-value", round(p_value,6))
        if p_value > alpha:
            print("\nDo not reject the null hypothesis")
            print("Mean of", name_0, 
                  "is not statistically different from",
                  "Mean of", name_1)
        else:
            print("\nReject the null hypothesis")
            print("Mean of", name_0, 
                  "is statistically different from",
                  "Mean of", name_1)
    
    # apply Mann-Whitney test if the normality assumption does not hold
    else:
        statistic, p_value = stats.mannwhitneyu(X_0,X_1)
        # print the Mann-Whitney test results
        print("\nMann-Whitney Test")
        print("Test statistic", round(statistic,6))
        print("p-value", round(p_value,6))
        if p_value > alpha:
            print("\nDo not reject the null hypothesis")
        else:
            print("\nReject the null hypothesis")  
    
    # present the boxplot for the data
    list_data = [X_0, X_1]
    plt.boxplot(list_data,labels=binary_categories)
    plt.show()
    

def two_means_paired(df, # pandas dataframe
                     target_1, # name of  variable 1
                     target_2, # name of variable 2
                     alpha = 0.05, # significance level
                     nan = 'propagate' # nan policy                     
                     ):
    
    # copy the data for two variables and difference
    df_1 = df[[target_1]].copy()
    df_2 = df[[target_2]].copy()
    D = df_1[target_1].sub(df_2[target_2],axis='index').values
    X_1 = df_1[target_1].values
    X_2 = df_2[target_2].values
    
    # get the main sample statistics for both variables and for difference
    statistics_quantitative(X_1,target_1)
    statistics_quantitative(X_2,target_2)
    statistics_quantitative(D,"Difference")
      
    # evaluate the normality assumption
    assumption = evaluate_normality_assumption(D, 'Difference',nan)
    
    if assumption == True:
        # get the inferential statistics
        print("\nINFERENTIAL STATISTICS")
        mean_1, var_1, std_1 = stats.mvsdist(X_1)
        mean_2, var_2, std_2 = stats.mvsdist(X_2)
        
        print("\nBayesian", str((1-alpha)*100)+"%",
              "confidence interval for the mean of", target_1,":",
              mean_1.interval(1-alpha))
        print("\nBayesian", str((1-alpha)*100)+"%",
              "confidence interval for the mean of", target_2,":",
              mean_2.interval(1-alpha))        
        print("\nBayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation of",
              target_1, ":",
              std_1.interval(1-alpha))
        print("\nBayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation of",
              target_2, ":",
              std_2.interval(1-alpha))
               
        print("\nPAIRED SAMPLES TEST OF MEAN")
        # get the test statistics and p-value
        t, p_value = stats.mstats.ttest_rel(X_1,X_2)
        # printout the test's main results
        if p_value > alpha:
            print("\nTest statistic", t)
            print("p-value",p_value)
            print("\nDo not reject the null hypothesis")
            print("Mean difference is not statistically significant")
        else:
            print("\nTest statistic", t)
            print("p-value",p_value)
            print("\nReject the null hypothesis")
            print("Mean difference is statistically significant")
    
    # apply Wilcoxon test if the normality assumption does not hold
    else:
        statistic, p_value = stats.wilcoxon(D)
        # print the Wilcoxon test results
        print("\nWilcoxon Test")
        print("Test statistic", round(statistic,6))
        print("p-value", round(p_value,6))
        if p_value > alpha:
            print("\nDo not reject the null hypothesis")
        else:
            print("\nReject the null hypothesis") 
    statistic, p_value = stats.wilcoxon(D)
    

def anova(df, # pandas dataframe
          target_name, # name of target variable
          category_name, # name category used to divide the samples
          categories_classes, # classes for category
          alpha = 0.05, # significance level
          nan = 'propagate' # nan policy
          ):
    
    X = [] # n-independent samples list
    num_variables = len(categories_classes) # number of independent samples
    assumptions = [] # list for normality assumption values
    
    
    for class_value in categories_classes:
        # sample for the class value
        new_df = df.loc[df[category_name]==class_value]
        # numpy array for category
        X_i = new_df[target_name].values
        # append array to samples
        X.append(X_i)
    
    # printout the main sample statistics and evaluate normality 
    for i in range (0,num_variables):
        X_i = X[i]
        name = categories_classes[i]
        statistics_quantitative(X_i,name)
        assumption = evaluate_normality_assumption(X_i,name,nan)
        assumptions.append(assumption)    
    
    # apply ANOVA if normality assumption holds for all groups
    if False not in assumptions:
        # get the main sample statistics
        print("\nINFERENTIAL STATISTICS")
        for i in range(0,num_variables):
            # get the Bayesian confidence intervals
            mean, var, std = stats.mvsdist(X[i])
            name = categories_classes[i]
            print("\nBayesian", str((1-alpha)*100)+"%",
              "confidence interval for the mean of", name,":",
              mean.interval(1-alpha))
        
            print("\nBayesian", str((1-alpha)*100)+"%", 
              "confidence interval for the standard deviation of",
              name, ":",
              std.interval(1-alpha))
     
        # apply the Levene and the t-test
        W, p_value_levene = stats.levene(*X)
        if p_value_levene > alpha:
            print("\nEqual variances assumed by Levene test")
            print("Significance of Levene test", round(p_value_levene,6))
            statistic, p_value = stats.f_oneway(*X)
            
        else:
            print("\nEqual variances not assumed by Levene test")
            print("Significance of Levene test", round(p_value_levene,6))
            statistic, p_value = stats.f_oneway(*X)
            statistic_KW, p_value_KW = stats.kruskal(*X)
        
        # output the ANOVA results
        print("\nANOVA")
        print("Test statistic", round(statistic,6))
        print("p-value", round(p_value,6))
        if p_value > alpha:
            print("\nDo not reject the null hypothesis")
        else:
            print("\nReject the null hypothesis")
        
        # If equality of variances is rejected
        if p_value_levene <= alpha:
            # print the Kruskal Wallis test results
            print("\nKruskal-Wallis Test")
            print("Test statistic", round(statistic_KW,6))
            print("p-value", round(p_value_KW,6))
            if p_value_KW > alpha:
                print("\nDo not reject the null hypothesis")
            else:
                print("\nReject the null hypothesis")            
    else:
        statistic_KW, p_value_KW = stats.kruskal(*X)
        # print the Kruskal Wallis test results
        print("\nKruskal-Wallis Test")
        print("Test statistic", round(statistic_KW,6))
        print("p-value", round(p_value_KW,6))
        if p_value_KW > alpha:
            print("\nDo not reject the null hypothesis")
        else:
            print("\nReject the null hypothesis")

    # present the boxplot for the data
    plt.boxplot(X,labels=categories_classes)
    plt.show()


def proportions_test(df, # pandas dataframe
                     binary, # binary variable name
                     name_0, # binary counted as failure
                     name_1, # binary counted as success
                     value_null, # value for the null hypothesis
                     profile, # two-sided, smaller, larger
                     alpha # significance level
                     ):
    
    df_0 = df.loc[df[binary]==name_0] # sample for first category
    df_1 = df.loc[df[binary]==name_1] # sample for second category
    num_0=len(df_0)
    num_1=len(df_1)
    N = num_0+num_1
    print("\nNumber of successes", num_1)
    print("Sample size",N)
    print("Sample proportion",round(num_1/N,6))
    stat, pval = proportions_ztest(count=num_1, nobs=N,value=value_null,
                                   alternative=profile)
    print("\nProportions test")
    print("Test statistic", round(stat,6))
    print("p-value", round(pval,6))
    if pval > alpha:
        print("Do not reject the null hypothesis")
        if profile=="two-sided":
            print("Success proportion is not statistically different from", 
                  value_null)
        elif profile=="smaller":
            print("Success proportion is not statistically smaller than", 
                  value_null)
        else:
            print("Success proportion is not statistically larger than", 
                  value_null)
            
    else:
        print("Reject the null hypothesis")
        if profile=="two-sided":
            print("Success proportion is statistically different from", 
                  value_null)
        elif profile=="smaller":
            print("Success proportion is statistically smaller than", 
                  value_null)
        else:
            print("Success proportion is statistically larger than", 
                  value_null)
    ci_low, ci_upp = proportion_confint(num_1, N, alpha, method='beta')
    print((1-alpha)*100, "% Clopper-Pearson confidence interval for proportion",
          "("+str(ci_low), str(ci_upp)+")")
    

def linear_regression_analysis(df, # pandas dataframe
                               predictor_name, # name of predictor
                               target_name # name of target
                               ):
    # copy the data for two variables and difference
    df_1 = df[[predictor_name]].copy()
    df_2 = df[[target_name]].copy()
    predictor = df_1[predictor_name].values
    target = df_2[target_name].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictor,target)
    
    print("\nRegression Results")
    print("\nR^2:", r_value**2)
    print("R:", r_value)
    print("\nIntercept:", intercept)
    print("Slope", slope)
    print("p-value of slope:", round(p_value,6))
    
    x=[]
    y=[]
    
    for i in range(int(np.min(predictor)),int(np.max(predictor))):
        x.append(i)
        y.append(i*slope+intercept)
    
    y=np.array(y)   
    y.reshape(-1, 1)
    
    plt.scatter(predictor, target)
    plt.scatter(x,y, c='k',marker='.')
    plt.xlabel(predictor_name)
    plt.ylabel(target_name)
    plt.show()
    

def classification_problem(data, # pandas dataframe
                           predictors, # predictors name list
                           target, # target name
                           architecture, # machine learning architecture
                           randomize_data=True, # data randomization
                           training_p=0.5, # training data proportion
                           seed=1, # seed for random number generator
                           n_shuffles=2, # number of times to shuffle
                           return_elements=False # return elements
                           ):
    # data sample randomization if asked (default)
    if randomize_data == True:
        for i in range(0,n_shuffles):
            data = data.sample(frac=1,random_state=seed).reset_index(drop=True)
    
    # get the predictors and target data
    X = data[predictors] # predictors data
    y = data[[target]] # target data
    
    # build the training and test samples:
    N = len(data) # number of data points
    N_train = int(training_p*len(data))
    X_train = X[:N_train]
    y_train = np.ravel(y[:N_train])
    X_test = X[N_train:N]
    y_test = np.ravel(y[N_train:N])
    
    print("\nNumber of data points:", N)
    print("Training data:", len(y_train))
    print("Test data:", len(y_test))
    
    # apply machine learning architecture:
    architecture.fit(X_train,y_train) # train the architecture
    y_pred_train = architecture.predict(X_train) # training set predictions
    y_pred_test = architecture.predict(X_test) # test set predictions
    
    # print the main metrics
    print("\nAccuracy scores")
    print("Training sample:", accuracy_score(y_train, y_pred_train))
    print("Test sample", accuracy_score(y_test,y_pred_test))
    
    # get the accuracy matrices for training and test data
    cnf_matrix_train = confusion_matrix(y_train, y_pred_train)
    cnf_matrix_test = confusion_matrix(y_test, y_pred_test)
    
    print("\nConfusion Matrices")
    print("\nTraining data")
    print(cnf_matrix_train)
    print("\nTest data")
    print(cnf_matrix_test)
    
    accuracy_matrix_train = []
    accuracy_matrix_test = []
    distance_train = 0
    distance_test = 0
    
    num_classes = len(cnf_matrix_train)
    
    # build the accuracy matrix and distance metric
    for i in range(0,num_classes):
        # get the number of training samples in target class i
        N_train = sum(cnf_matrix_train[i])
        # get the number of test samples in target class i
        N_test = sum(cnf_matrix_test[i])
        # new lines for training and testing accuracies
        new_line_train = []
        new_line_test = []
        # get the accuracies and distance for class i
        for j in range(0,num_classes):
            # get the accuracy scores for training and testing
            p_ij_train = cnf_matrix_train[i][j]/N_train
            p_ij_test = cnf_matrix_test[i][j]/N_test
            # add the new accuracy score for pair (i,j)
            new_line_train.append(p_ij_train)
            new_line_test.append(p_ij_test)
            # for a diagonal square (when i = j)
            if i == j:
                # calculate training and testing squared distances for 
                # this class and add to the metrics scores
                distance_train += (p_ij_train-1)**2
                distance_test += (p_ij_test-1)**2
            
        # add the accuracies for class i to the accuracy matrix
        accuracy_matrix_train.append(new_line_train)
        accuracy_matrix_test.append(new_line_test)
    
    # change the accuracy matrices from list to array format
    accuracy_matrix_train = np.asfarray(accuracy_matrix_train)
    accuracy_matrix_test = np.asfarray(accuracy_matrix_test)
    
    # extract the training and test distance using the square root
    distance_train = np.sqrt(distance_train)
    distance_test = np.sqrt(distance_test)
    
    # printout the accuracy matrices and distances for training
    # test data (and calibration data)
    print("\nAccuracy Matrices:")
    print("\nTraining data:")
    print(accuracy_matrix_train)
    print("\nDistance:", distance_train)
    print("\nTest data:")
    print(accuracy_matrix_test)
    print("\nDistance:", distance_test)
    
    # return training data, test data with predictions if requested
    if return_elements == True:
        training_data_list = [X_train, y_train, y_pred_train]
        test_data_list = [X_test, y_test, y_pred_test]
        return training_data_list, test_data_list
    
    
def regression_problem(data, # pandas dataframe
                           predictors, # predictors name list
                           target, # target name
                           architecture, # machine learning architecture
                           randomize_data=True, # data randomization
                           training_p=0.5, # training data proportion
                           seed=1, # seed for random number generator
                           n_shuffles=2, # number of times to shuffle
                           return_elements=False # return elements
                           ):
    # data sample randomization if asked (default)
    if randomize_data == True:
        for i in range(0,n_shuffles):
            data = data.sample(frac=1,random_state=seed).reset_index(drop=True)
    
    # get the predictors and target data
    X = data[predictors] # predictors data
    y = data[[target]] # target data
    
    # build the training and test samples:
    N = len(data) # number of data points
    N_train = int(training_p*len(data))
    X_train = X[:N_train]
    y_train = np.ravel(y[:N_train])
    X_test = X[N_train:N]
    y_test = np.ravel(y[N_train:N])
    
    print("\nNumber of data points:", N)
    print("Training data:", len(y_train))
    print("Test data:", len(y_test))
    
    # apply machine learning architecture:
    architecture.fit(X_train,y_train) # train the architecture
    y_pred_train = architecture.predict(X_train) # training set predictions
    y_pred_test = architecture.predict(X_test) # test set predictions
    
    # print the main metrics
    print("\nRMSE")
    print("Training sample:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print("Test sample", np.sqrt(mean_squared_error(y_test,y_pred_test)))
    print("\nMean Absolute Error")
    print("Training sample:", mean_absolute_error(y_train, y_pred_train))
    print("Test sample", mean_absolute_error(y_test,y_pred_test))
    print("\nMedian Absolute Error")
    print("Training sample:", median_absolute_error(y_train, y_pred_train))
    print("Test sample", median_absolute_error(y_test,y_pred_test))
    
    # return training data, test data with predictions if requested
    if return_elements == True:
        training_data_list = [X_train, y_train, y_pred_train]
        test_data_list = [X_test, y_test, y_pred_test]
        return training_data_list, test_data_list

    
def inference_proportions(df, # pandas dataframe
                          binary, # binary variable name
                          name_1, # binary counted as 1
                          alpha=0.05
                          ):
    # return Clopper-Pearson confidence interval for proportion
    df_1 = df.loc[df[binary]==name_1]
    num_1=len(df_1)
    N=len(df)
    ci_low, ci_upp = proportion_confint(num_1, N, alpha, method='beta')
    interval = (ci_low, ci_upp)
    return interval


def inference_target_variable(df, # pandas dataframe
                              target_name, # target variable name
                              alpha=0.05, # significance level
                              ):
    # return the confidence intervals for the mean and standard deviation
    # of a target variable
    X=df[target_name].values
    mean, var, std = stats.mvsdist(X)
    return mean.interval(1-alpha), std.interval(1-alpha)
    

def inference_target_category(df, # pandas dataframe
                              target_variable, # target variable name
                              target_category, # target category name
                              binary_variable, # binary variable name
                              alpha=0.05 # significance level
                              ):
    # return the confidence intervals for the mean and standard deviation
    # of a target variable
    df_1 = df.loc[df[binary_variable]==target_category]
    X = df_1[target_variable].values
    mean, var, std = stats.mvsdist(X)
    return mean.interval(1-alpha), std.interval(1-alpha)

def get_category(df, # pandas dataframe
                 category_variable_name, # category variable name
                 category_value): # category value
    # return a dataframe from a parent dataframe with a specific value for a
    # category variable
    return df.loc[df[category_variable_name]==category_value]


