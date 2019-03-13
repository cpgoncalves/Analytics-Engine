# Analytics-Engine

ON THE SOFTWARE MODULE

Analytics Engine is an AI-based system with data analysis automation built that works with pandas, stats, statsmodels and scikit-learn, it was  developed to work as a Business Analytics for Business Intelligence module, as part of classroom material, in the context of Statistics of Management classes.

The system automates assumptions testing and, in the case of the parametric tests, it is not intended as just presenting the test but also providing descriptive and inference elements that may support the Business Intelligence analyst in his analysis and report, this includes not only automating many of the different steps that one must take to apply a parametric test but also reading out the test results and providing additional statistical elements that may support the analyst.

The Machine Learning functionalities automates some of the common steps involved in applying scikit-learn to business problems.

The software was intended to be used as part of undergraduate classroom material in teaching of Statistics for Management in the context of Business Analytics as a support to Business Intelligence and Corporate Decision Making. The module can be used in conjunction with Python's major data science functionalities and also support professional work and research in the field of statistics for Business Intelligence.


ON THE MAIN FUNCTIONS

Normality Test:

evaluate_normality_assumption(X,target_name,nan='propagate'), returns True if the assumption holds and False if it does not hold at a 1% significance level.

Main sample statistics for a quantitative variable:
statistics_quantitative(X,target_name), returns the main sample statistics along with the skewness and kurtosis p-values

One Sample Mean and Dispersion Analysis

one_mean(df, # pandas dataframe
        target_name, # target variable name
        popmean, # population mean for used for null hypothesis
        alpha=0.05, # significance level
        nan='propagate' # nan policy
        )

The one sample Test of mean provides the main sample statistics plus boxplot, automatically tests the normality assumption and, if the assumption holds, it provides the Bayesian confidence intervals for the mean and the standard deviation and also performs the one sample t-test for the given population mean and significance level, reading out the results for further analysis by the Business Intelligence analyst.


Two Independent Samples Means and Dispersion Analysis

two_means_indep(df, # pandas dataframe
                target_name, # name of target variable
                binary_variable, # binary variable name
                binary_categories, # binary variable categories
                alpha = 0.05, # significance level
                nan = 'propagate' # nan policy
                )

The two independent samples means and dispersion analysis functionality provides the main sample statistics plus boxplot for the two independent samples, automatically tests the normality assumption and, if the assumption holds, it provides the confidence intervals for the mean and the standard deviation of each sample and also performs the Levene's test for equality of variances and the independent samples t-test for the equality of means, the later with the p-value already adapted to the equality of variances assumption. If the normality assumption does not hold, the system automatically performs the Mann-Whitney test of equality of distributions as the nonparametric alternative.


Paired-Samples Mans and Dispersion Analysis




