import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd


###Numeric data
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]

x
x_with_nan

###Replace missing values with nan
math.isnan(np.nan), np.isnan(math.nan)

math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

###Create arrays/series

y, y_with_nan = np.array(x), np.array(x_with_nan)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

y
z

y_with_nan

z_with_nan


###Calculate the mean with & w/o built-in python statistics functions
#W/O
mean_ = sum(x) / len(x)
mean_

#W
mean_ = statistics.mean(x)
mean_
mean_ = statistics.fmean(x)
mean_


###Trying to obtain the mean but with nan values
mean_ = statistics.mean(x_with_nan)
mean_

mean_ = statistics.fmean(x_with_nan)
mean_

###Obtaining the mean with numpy
mean_ = np.mean(y)
mean_

###Mean with .mean() function
mean_ = y.mean()
mean_

##Mean using numpy & statistics function for nan values
np.mean(y_with_nan)

y_with_nan.mean()

###Ignores nan values when obtaining mean
np.nanmean(y_with_nan)

mean_ = z.mean()
mean_

##Using pandas to ignore nan values for mean
z_with_nan.mean()



###Weighted mean
0.2 * 2 + 0.5 * 4 + 0.3 * 8


##Weighted mean with sum()
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

##Using numpy to get weighted mean
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

wmean = np.average(z, weights=w)
wmean

##Using .sum() for weighted mean
(w * y).sum() / w.sum()

##Weighted mean with nan values
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)


###Harmonic Mean
hmean = len(x) / sum(1 / item for item in x)
hmean


hmean = statistics.harmonic_mean(x)
hmean


##Harmonic mean for nan, 0, & statistics error
statistics.harmonic_mean(x_with_nan)

statistics.harmonic_mean([1, 0, 2])

statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError - Does not support negative values

##Using scipy for harmonic mean
scipy.stats.hmean(y)

scipy.stats.hmean(z)


###Geometric mean
gmean = 1
for item in x:
    gmean *= item

gmean **= 1 / len(x)
gmean

##Converting values to floating-point numbers for geometric mean
gmean = statistics.geometric_mean(x)
gmean

##Geometric mean with nan values
gmean = statistics.geometric_mean(x_with_nan)
gmean

##Scipy for geometric mean
scipy.stats.gmean(y)

scipy.stats.gmean(z)






####Median
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_

##Statistics function for median
median_ = statistics.median(x)
median_

median_ = statistics.median(x[:-1])
median_

##Median low/high
statistics.median_low(x[:-1])

statistics.median_high(x[:-1])


##Median with nan

statistics.median(x_with_nan)

statistics.median_low(x_with_nan)

statistics.median_high(x_with_nan)


##Numpy for median
median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

##Numpy for median with nan
np.nanmedian(y_with_nan)

np.nanmedian(y_with_nan[:-1])

##Ignoring nan values with pandas for median
z.median()

z_with_nan.median()



###Mode
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_


##Using statistics funciton for mode
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_

##Multimode to reutrn list with all modes
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)


##Mode with nan
statistics.mode([2, math.nan, 2])

statistics.multimode([2, math.nan, 2])

statistics.mode([2, math.nan, 0, math.nan, 5])

statistics.multimode([2, math.nan, 0, math.nan, 5])

##Using scipy for mode
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

mode_ = scipy.stats.mode(v)
mode_


##Numpy arrays for mode
mode_.mode

mode_.count

##Pandas for multimodal values
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()

v.mode()

w.mode()



###Variance
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

##Using statistics function for variance
var_ = statistics.variance(x)
var_

##statistics function with nan for variance
statistics.variance(x_with_nan)

##Using numpy for variance
var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

##Using numpy with nan for variance
np.var(y_with_nan, ddof=1)

y_with_nan.var(ddof=1)


##Skip values
np.nanvar(y_with_nan, ddof=1)

##Ignoring nan values with python for variance
z.var(ddof=1)

z_with_nan.var(ddof=1)


###Standard deviation
std_ = var_ ** 0.5
std_

##Using statistics function for st dev
std_ = statistics.stdev(x)
std_

##nan values for st dev
np.std(y, ddof=1)

y.std(ddof=1)

np.std(y_with_nan, ddof=1)

y_with_nan.std(ddof=1)

np.nanstd(y_with_nan, ddof=1)

##pandas to skip nan for st dev
z.std(ddof=1)

z_with_nan.std(ddof=1)


###Skewness
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_

##Using scipy for skewness
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)

scipy.stats.skew(y_with_nan, bias=False)

##Using pandas for skewness
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()

z_with_nan.skew()


###Percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')


##Using numpy to get percentiles
y = np.array(x)
np.percentile(y, 5)

np.percentile(y, 95)

np.percentile(y, [25, 50, 75])

np.median(y)

##Ignoring nan values
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan

np.nanpercentile(y_with_nan, [25, 50, 75])

##Quantiles between 0 and 1
np.quantile(y, 0.05)

np.quantile(y, 0.95)

np.quantile(y, [0.25, 0.5, 0.75])

np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

##Using pandas for quantiles
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)

z.quantile(0.95)

z.quantile([0.25, 0.5, 0.75])

z_with_nan.quantile([0.25, 0.5, 0.75])


###Range
np.ptp(y)

np.ptp(z)

np.ptp(y_with_nan)

np.ptp(z_with_nan)

##Min/max
np.amax(y) - np.amin(y)

np.nanmax(y_with_nan) - np.nanmin(y_with_nan)

y.max() - y.min()

z.max() - z.min()

z_with_nan.max() - z_with_nan.min()

##Interquartile range
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]


###Summary of descriptive statistics
result = scipy.stats.describe(y, ddof=1, bias=False)
result

result.nobs

result.minmax[0]  # Min

result.minmax[1]  # Max

result.mean

result.variance

result.skewness

result.kurtosis

##Using python for results
result = z.describe()
result

###Percentile results
result['mean']

result['std']

result['min']

result['max']

result['25%']

result['50%']

result['75%']


###Correlation
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)


###Covariance
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

##covariance matrix with numpy
cov_matrix = np.cov(x_, y_)
cov_matrix

x_.var(ddof=1)

y_.var(ddof=1)

##Covariance between x and y 
cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy

##Using pandas for covariance
cov_xy = x__.cov(y__)
cov_xy

cov_xy = y__.cov(x__)
cov_xy


###Correlation coefficient
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

##Using scipy for correlation coefficient
r, p = scipy.stats.pearsonr(x_, y_)
r

p


###Correlation coefficient matrix
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

##Correlation coefficient between x and y
r = corr_matrix[0, 1]
r

r = corr_matrix[1, 0]
r

##Using scipy for correlation coefficient/linear regression
scipy.stats.linregress(x_, y_)

result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

##Using pandas for correlation coefficient
r = x__.corr(y__)
r

r = y__.corr(x__)
r



### 2D numpy array 
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a

np.mean(a)

a.mean()

np.median(a)

a.var(ddof=1)


##Numpy mean with different axis 
np.mean(a, axis=0)

a.mean(axis=0)

np.mean(a, axis=1)

a.mean(axis=1)

np.median(a, axis=0)

np.median(a, axis=1)

a.var(axis=0, ddof=1)

a.var(axis=1, ddof=1)


##Scipy with different axis
scipy.stats.gmean(a)  # Default: axis=0

scipy.stats.gmean(a, axis=0)

scipy.stats.gmean(a, axis=1)

scipy.stats.gmean(a, axis=None)


###2D Scipy
scipy.stats.describe(a, axis=None, ddof=1, bias=False)

scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0

scipy.stats.describe(a, axis=1, ddof=1, bias=False)


result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean


###Dataframes
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df


##DF mean/variance
df.mean()

df.var()

df.mean(axis=1)

df.var(axis=1)

df['A']

df['A'].mean()

df['A'].var()


df.values


df.to_numpy()

df.describe()

df.describe().at['mean', 'A']

df.describe().at['50%', 'B']