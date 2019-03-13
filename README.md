# Analytics Engine

ON THE SOFTWARE MODULE

Analytics Engine is an AI-based system with data analysis automation built that works with pandas, stats, statsmodels and scikit-learn, it was  developed to work as a Business Analytics for Business Intelligence module, as part of classroom material, in the context of Statistics of Management classes.

The system automates assumptions testing and, in the case of the parametric tests, it is not intended as just presenting the test but also providing descriptive and inference elements that may support the Business Intelligence analyst in his analysis and report, this includes not only automating many of the different steps that one must take to apply a parametric test but also reading out the test results given the significance level defined by the user and providing additional statistical elements that may support the analyst.

The Machine Learning functionalities automates some of the common steps involved in applying scikit-learn to business problems.

The software was intended to be used as part of undergraduate classroom material in teaching of Statistics for Management in the context of Business Analytics as a support to Business Intelligence and Corporate Decision Making. The module can be used in conjunction with Python's major data science functionalities and also support professional work and research in the field of statistics for Business Intelligence.


ON THE MAIN FUNCTIONS

1. Normality Test:

evaluate_normality_assumption(X,target_name,nan='propagate'), returns True if the assumption holds and False if it does not hold at a 1% significance level.

Main sample statistics for a quantitative variable:
statistics_quantitative(X,target_name), prints the main sample statistics along with the skewness and kurtosis p-values

2. One Sample Mean and Dispersion Analysis

one_mean(df, # pandas dataframe
        target_name, # target variable name
        popmean, # population mean for used for null hypothesis
        alpha=0.05, # significance level
        nan='propagate' # nan policy
        )

The one sample Test of mean provides the main sample statistics plus boxplot, automatically tests the normality assumption and, if the assumption holds, it provides the Bayesian confidence intervals for the mean and the standard deviation and also performs the one sample t-test for the given population mean and significance level, reading out the results for further analysis by the Business Intelligence analyst.


3. Two Independent Samples Means and Dispersion Analysis

two_means_indep(df, # pandas dataframe
                target_name, # name of target variable
                binary_variable, # binary variable name
                binary_categories, # binary variable categories
                alpha = 0.05, # significance level
                nan = 'propagate' # nan policy
                )

The two independent samples means and dispersion analysis functionality provides the main sample statistics plus boxplot for the two independent samples, automatically tests the normality assumption and, if the assumption holds, it provides the confidence intervals for the mean and the standard deviation of each sample and also performs the Levene's test for equality of variances and the independent samples t-test for the equality of means, the later with the p-value already adapted to the equality of variances assumption. If the normality assumption does not hold, the system automatically performs the Mann-Whitney test of equality of distributions as the nonparametric alternative.


4. Paired Samples Means and Dispersion Analysis

two_means_paired(df, # pandas dataframe
                     target_1, # name of  variable 1
                     target_2, # name of variable 2
                     alpha = 0.05, # significance level
                     nan = 'propagate' # nan policy                     
                     )

The paired samples means and dispersion analysis functionality provides the main sample statistics for two target variables plus for the difference between target variable 1 and target variable 2 (X1 - X2), automatically tests the normality assumption and, if the assumption holds, it provides the confidence intervals for the mean and the standard deviation of each variable and performs the paired samples t-test for equality of means. If the normality assumption does not hold, the system automatically performs the Wilcoxon test.

5. Multiple Independent Samples Means and Dispersion Analysis

anova(df, # pandas dataframe
      target_name, # name of target variable
      category_name, # name category used to divide the samples
      categories_classes, # classes for category
      alpha = 0.05, # significance
      nan = 'propagate' # nan policy
      )

The anova function implements a more than two independent samples and dispersion analysis providing the main sample statistics for each group plus boxplot, automatically tests the normality assumption and, if the assumption holds, it provides the confidence intervals for the mean and the standard deviation of each independent sample and also performs the Levene's test for equality of variances, if the Levene's test holds, the system applies the ANOVA test of equality of means, if it does not hold it outputs both the ANOVA and the Kruskal-Wallis test for equality of distributions. If the normality assumption does not hold, the system only performs the Kruskal-Wallis test for equality of distributions test as the nonparametric alternative to the ANOVA test.

4. Paired Samples Means and Dispersion Analysis


proportions_test(df, # pandas dataframe
                 binary, # binary variable name
                 name_0, # binary counted as failure
                 name_1, # binary counted as success
                 value_null, # value for the null hypothesis
                 profile, # two-sided, smaller, larger
                 alpha
                     )

The proportions test processes a pandas variable that is dichotomic, which can contain numeric or string lables, the user provides the binary category that is counted as failure and the category that id counted as success, the null hypothesis value for the proportion and the profile for the proportions z-test "two-sided" (p=p0), "smaller" (p<p0) or "larger" (p>p0). The system automatically outputs the number of successes, the sample size, the sample proportion, the proportions z test results and the  Clopper-Pearson confidence interval for the proportion.


5. Linear Regression Analysis

linear_regression_analysis(df, # pandas dataframe
                           predictor_name, # name of predictor
                           target_name # name of target
                           )

This is the basic linear regression for a target versus a single predictor with the dispersion plot and fitted regression line.


6. Classification Problem

classification_problem(data, # pandas dataframe
                       predictors, # predictors name list
                       target, # target name
                       architecture, # machine learning architecture
                       randomize_data=True, # data randomization
                       training_p=0.5, # training data proportion
                       seed=1, # seed for random number generator
                       n_shuffles=2, # number of times to shuffle
                       return_elements=False # return elements
                       )

This is a basic function that applies machine learning to a classification problem using scikit-learn functionalities. A pandas dataframe is provided by the user, the predictor variables names list as well as the target variable name, the architecture is also provided by the user. By default the system randomizes the sample with two shuffles using the provided random number generator seed, the user can also apply a single shuffle or no randomization at all if the randomize data is set to False. The user defines the proportion of data to be used for training, with the remaining data being set aside for testing. If the option return elements is set to True (the default is False), then, the function returns the trained machine learning architecture, the training data and the test data.

The function outputs the total sample size, the training data size, the test data size, the prediction accuracy scores on the training and test data, the number of classes in the target and the prediction accuracy matrices build from the confusion matrix containing the percentage of correct predictions for both training and testing, as well as the distance of these two matrices from the identity matrix (which would represent perfect score in prediction for each class of the target.

7. Regression Problem

regression_problem(data, # pandas dataframe
                   predictors, # predictors name list
                   target, # target name
                   architecture, # machine learning architecture
                   randomize_data=True, # data randomization
                   training_p=0.5, # training data proportion
                   seed=1, # seed for random number generator
                   n_shuffles=2 # number of times to shuffle
                   )

The regression problem functionality is another machine learning function that is applied analogously to the classification_problem function but for a problem of regression instead of classification. Instead of the accuracies it produces for both training and test samples the Root Mean Square Error (RMSE), the Mean Absolute Error and the Median Absolute Error.







                           




