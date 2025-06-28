# Formula Reference

Short English description of every supported formula.

---

## Date/Time

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `DATE` | Create date value | year, month, day | `=DATE(2025, 6, 27)` |
| `DAY` | Day component | date | `=DAY("2025-06-27")` |
| `DAYS` | Days between dates | end_date, start_date | `=DAYS("2025-06-27", "2025-06-01")` |
| `HOUR` | Hour component | datetime | `=HOUR("2025-06-27 14:30")` |
| `MINUTE` | Minute component | datetime | `=MINUTE("2025-06-27 14:30")` |
| `MONTH` | Month component | date | `=MONTH("2025-06-27")` |
| `NETWORKDAYS` | Working days between dates | start_date, end_date | `=NETWORKDAYS("2025-06-01", "2025-06-27")` |
| `NOW` | Current date and time | — | `=NOW()` |
| `SECOND` | Second component | datetime | `=SECOND("2025-06-27 14:30:45")` |
| `TODAY` | Current date | — | `=TODAY()` |
| `WEEKDAY` | Day of week | date, type? | `=WEEKDAY("2025-06-27")` |
| `WEEKNUM` | Week number | date | `=WEEKNUM("2025-06-27")` |
| `YEAR` | Year component | date | `=YEAR("2025-06-27")` |
| `EDATE(date, months)` | Date shifted by N months | start_date, months | `=EDATE("2025-01-15", -3)` |
| `EOMONTH(date, months?)` | End-of-month date after offset | start_date, months? | `=EOMONTH("2025-01-10", 1)` |
| `WORKDAY(date, days)` | Date after N workdays (Mon–Fri) | start_date, days | `=WORKDAY("2025-06-25", 7)` |
| `YEAR` | Year component | date | `=YEAR("2025-06-27")` |

## Financial

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `ALPHA` | Alpha coefficient | returns_range, benchmark_range | `=ALPHA(B2:B252, C2:C252)` |
| `BETA` | Beta coefficient | returns_range, benchmark_range | `=BETA(B2:B252, C2:C252)` |
| `CVAR` | Conditional Value at Risk | returns_range, confidence_level | `=CVAR(B2:B252, 0.95)` |
| `EPS` | Earnings per Share | net_income, shares_outstanding | `=EPS(3500000, 1000000)` |
| `ROA` | Return on Assets | net_income, total_assets | `=ROA(350000, 5000000)` |
| `ROE` | Return on Equity | net_income, shareholder_equity | `=ROE(350000, 2000000)` |
| `SHARPE` | Sharpe ratio | returns_range, risk_free_rate | `=SHARPE(B2:B252, 0.02)` |

## Logical

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `AND` | Logical AND | logical1, logical2, … | `=AND(A1>0, B1<5)` |
| `IF` | Return a if condition else b | condition, value_if_true, value_if_false | `=IF(A1>0, "Yes", "No")` |
| `IFERROR` | Return alternate if error | value, alternate | `=IFERROR(1/0, "error")` |
| `IFNA(value, alt)` | Returns alt if value is #N/A; otherwise returns value | value, alt | `=IFNA(VLOOKUP("Bob", A:A, 1), "N/A")` |
| `IFS` | Multiple conditions | condition1, value1, condition2, value2, … | `=IFS(A1>90, "A", A1>80, "B", TRUE, "C")` |
| `NOT` | Logical NOT | logical | `=NOT(A1>0)` |
| `OR` | Logical OR | logical1, logical2, … | `=OR(A1>0, B1<5)` |
| `XOR` | Logical exclusive OR | logical1, logical2, … | `=XOR(A1>0, B1<5)` |

## Lookup

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `CHOOSE(index, option1, option2, …)` | Returns the value at position index (1-based) from the list of options | index, option1, option2, … | `=CHOOSE(2, "red", "green", "blue")` |
| `COLUMN` | Column number of reference | cell_reference? | `=COLUMN(B3)` |
| `COUNTUNIQUE` | Count unique values | range | `=COUNTUNIQUE(A1:A100)` |
| `UNIQUE` | Unique values from range | range | `=UNIQUE(A1:A10)` |

## Math

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `ABS` | Absolute value | number | `=ABS(-5)` |
| `AVERAGE` | Mean of values | numbers/range | `=AVERAGE(B1:B5)` |
| `CEILING` | Round up to multiple | number, multiple | `=CEILING(7.2, 1)` |
| `COUNT` | Count numeric cells | numbers/range | `=COUNT(A1:A10)` |
| `EVEN` | Round up to even | number | `=EVEN(5.3)` |
| `FACT` | Factorial | integer | `=FACT(5)` |
| `FLOOR` | Round down to multiple | number, multiple | `=FLOOR(7.9, 1)` |
| `GCD(a, b, …)` | Greatest common divisor of one or more integers | integers | `=GCD(12, 18)` |
| `INT` | Integer part of number | number | `=INT(3.7)` |
| `LCM(a, b, …)` | Least common multiple of integers | integers | `=LCM(3, 4, 5)` |
| `LN` | Natural log | number | `=LN(10)` |
| `LOG` | Logarithm base x | number, base | `=LOG(100, 10)` |
| `LOG10` | Base-10 logarithm | number | `=LOG10(1000)` |
| `MAX` | Maximum value | numbers/range | `=MAX(C1:C20)` |
| `MIN` | Minimum value | numbers/range | `=MIN(C1:C20)` |
| `MOD` | Remainder of division | dividend, divisor | `=MOD(10, 3)` |
| `ODD` | Round up to odd | number | `=ODD(6)` |
| `PI` | Pi constant | — | `=PI()` |
| `POW` | Power x^y | base, exponent | `=POW(2, 3)` |
| `PRODUCT` | Product of values | numbers/range | `=PRODUCT(A1:A3, B1:B3)` |
| `QUOTIENT` | Integer division | dividend, divisor | `=QUOTIENT(10, 3)` |
| `RANK` | Rank of number in list | number, range | `=RANK(85, B1:B20)` |
| `ROUND` | Round to n decimals | number, decimals | `=ROUND(3.1416, 2)` |
| `ROUNDDOWN` | Round down (toward zero) | number, decimals | `=ROUNDDOWN(3.99, 0)` |
| `ROUNDUP` | Round up (away from zero) | number, decimals | `=ROUNDUP(3.14, 1)` |
| `SIGN` | Sign of number | number | `=SIGN(-7)` |
| `SQRT` | Square root | number | `=SQRT(16)` |
| `SUM` | Sum of values | numbers/range | `=SUM(A1:A3)` |
| `TRUNC` | Truncate decimals | number, decimals | `=TRUNC(3.1416, 2)` |

## Other

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `ABSMAX(a, b)` | Returns the larger absolute value of the two numbers | numbers | `=ABSMAX(-3, 5)` |
| `ACOSH(x)` | Inverse hyperbolic cosine | number | `=ACOSH(2)` |
| `AIC(k, LL)` | Akaike Information Criterion: 2k − 2LL | k, log-likelihood | `=AIC(5, -123)` |
| `AR_MEAN(range)` | Arithmetic mean (simple average) of values | numbers/range | `=AR_MEAN(B2:B10)` |
| `ASINH(x)` | Inverse hyperbolic sine | number | `=ASINH(1.5)` |
| `ATANH(x)` | Inverse hyperbolic tangent | number | `=ATANH(0.5)` |
| `ATR_VAL(high, low, close, period)` | Average True Range over the period | highs, lows, closes, period | `=ATR_VAL(B2:B15, C2:C15, D2:D15, 14)` |
| `BIC(k, LL, n)` | Bayesian Information Criterion | k, log-likelihood, n | `=BIC(5, -123, 100)` |
| `BITAND(a, b)` | Bitwise AND of two 32-bit integers | integers | `=BITAND(5, 3)` |
| `BITNOT(x)` | Bitwise NOT of 32-bit integer | integer | `=BITNOT(5)` |
| `BITOR(a, b)` | Bitwise OR of two 32-bit integers | integers | `=BITOR(5, 1)` |
| `BITXOR(a, b)` | Bitwise exclusive-OR of two 32-bit integers | integers | `=BITXOR(5, 1)` |
| `BOLLWIDTH(series, window, k)` | Relative width of Bollinger Bands ( (upper − lower)/MA ) | series, window, k | `=BOLLWIDTH(B2:B101, 20, 2)` |
| `CHAIKIN_MF(high, low, close, volume)` | Chaikin Money Flow indicator of buying vs selling pressure | high, low, close, volume | `=CHAIKIN_MF(B2:B101, C2:C101, D2:D101, E2:E101)` |
| `CHEBYSHEV(n, x)` | Chebyshev polynomial Tₙ(x) | n, x | `=CHEBYSHEV(3, 0.5)` |
| `CHISQ_P(observed, expected)` | p-value from Chi-square test | observed_range, expected_range | `=CHISQ_P(B2:B5, C2:C5)` |
| `CLAMP(x, min, max)` | Constrain x to the range [min, max] | x, min, max | `=CLAMP(1.2, 0, 1)` |
| `CLAMP01(x)` | Clamp x into the range [0, 1] | x | `=CLAMP01(1.5)` |
| `CLIP(x, min, max)` | Limit x to the closed interval [min, max] | x, min, max | `=CLIP(-2, 0, 10)` |
| `COEFFICIENT_VAR(range)` | Coefficient of variation = stdev / mean | numbers/range | `=COEFFICIENT_VAR(B2:B20)` |
| `COEFF_DET(y_true, y_pred)` | Coefficient of determination R² | actual_range, predicted_range | `=COEFF_DET(C2:C20, D2:D20)` |
| `COGS` | Cost of Goods Sold (business accounting) | units_sold, unit_cost | `=COGS(100, 5.75)` |
| `COMBIN(n, k)` | Number of combinations "n choose k" | n, k | `=COMBIN(10, 3)` |
| `CORREL(x_range, y_range)` | Pearson correlation coefficient | x_range, y_range | `=CORREL(A2:A20, B2:B20)` |
| `COSINE_SIM(a, b)` | Cosine similarity between two vectors | vector_a, vector_b | `=COSINE_SIM(A1:E1, A2:E2)` |
| `COTH(x)` | Hyperbolic cotangent cosh(x) / sinh(x) | x | `=COTH(1.2)` |
| `COUNTA` | Count non-empty cells | range | `=COUNTA(A1:A10)` |
| `COUNTBLANK(range)` | Counts empty cells in the specified range | range | `=COUNTBLANK(A1:A10)` |
| `COVAR(x_range, y_range)` | Covariance between two datasets | x_range, y_range | `=COVAR(A2:A10, B2:B10)` |
| `CSCH(x)` | Hyperbolic cosecant 1 / sinh(x) | x | `=CSCH(2)` |
| `CUBE(x)` | x³ (number cubed) | x | `=CUBE(4)` |
| `CUMPROD(range)` | Cumulative product running down the list | numbers/range | `=CUMPROD(B2:B10)` |
| `DDB(cost, salvage, life, period)` | Double-declining balance depreciation for the given period | cost, salvage, life, period | `=DDB(10000, 1000, 5, 2)` |
| `DIFF(range)` | First difference: current minus previous element | series/range | `=DIFF(B2:B20)` |
| `DOT(a, b)` | Dot product Σ aᵢ bᵢ of two equal-length vectors | vector_a, vector_b | `=DOT(A1:A3, B1:B3)` |
| `DURBINWATSON(resid)` | Durbin–Watson statistic to detect autocorrelation in regression residuals | residuals | `=DURBINWATSON(C2:C100)` |
| `EMA(series, period)` | Exponential moving average over the specified period | series, period | `=EMA(B2:B100, 20)` |
| `ENTROPY(range)` | Shannon entropy (information content) in bits | numbers/range | `=ENTROPY(C2:C50)` |
| `EUCLIDEAN(a, b)` | Euclidean distance between two equal-length vectors | vector_a, vector_b | `=EUCLIDEAN(A1:A3, B1:B3)` |
| `EXP(x)` | Natural exponential e^x | x | `=EXP(1)` |
| `EXPM1(x)` | e^x − 1 with higher precision for small x | x | `=EXPM1(0.01)` |
| `EXP_SMOOTH(series, α)` | Exponential smoothing forecast with smoothing factor α | series, alpha | `=EXP_SMOOTH(B2:B50, 0.2)` |
| `FACTDOUBLE(n)` | Double factorial n!! (product of integers with same parity) | integer | `=FACTDOUBLE(7)` |
| `FORCE_INDEX(close, volume, period)` | Price momentum indicator: change in price × volume, averaged over period | close_prices, volumes, period | `=FORCE_INDEX(C2:C101, D2:D101, 13)` |
| `FREQUENCY(data, bins)` | Histogram counts of data falling into each bin interval | data_range, bins_range | `=FREQUENCY(A2:A20, B2:B6)` |
| `GEOMEAN(range)` | Geometric mean of positive numbers | numbers/range | `=GEOMEAN(C2:C10)` |
| `GINI(range)` | Gini coefficient measuring inequality (0 = perfect equality, 1 = max) | numbers/range | `=GINI(C2:C100)` |
| `GPM(revenue, cogs)` | Gross profit margin = (revenue − COGS) ⁄ revenue | revenue, cogs | `=GPM(10000, 7000)` |
| `HAMMING(a, b)` | Hamming distance – count of positions where two strings/vectors differ | vector_a, vector_b | `=HAMMING(A1:E1, A2:E2)` |
| `HARMEAN(range)` | Harmonic mean of positive numbers | numbers/range | `=HARMEAN(B2:B10)` |
| `HASH(text, algo)` | Cryptographic hash of text using algorithm (e.g., SHA-256) | text, algorithm | `=HASH("hello", "sha256")` |
| `HURST(series)` | Hurst exponent estimating long-term memory of a time series | series | `=HURST(B2:B200)` |
| `HYPOT(x, y)` | Euclidean norm √(x² + y²) | x, y | `=HYPOT(3, 4)` |
| `INTERCEPT(y_range, x_range)` | y-intercept of the best-fit regression line | y_range, x_range | `=INTERCEPT(C2:C20, B2:B20)` |
| `IQR(range)` | Inter-quartile range (Q3 − Q1) | numbers/range | `=IQR(C2:C20)` |
| `JACCARD(a, b)` | Jaccard similarity = \|A ∩ B\| ⁄ \|A ∪ B\| | set_a, set_b | `=JACCARD(A1:A10, B1:B10)` |
| `KS_TEST_P(sample, cdf)` | p-value from the Kolmogorov–Smirnov goodness-of-fit test | sample_range, cdf | `=KS_TEST_P(B2:B50, "normal")` |
| `L1NORM(range)` | Manhattan norm – sum of absolute values | numbers/range | `=L1NORM(C2:C10)` |
| `L2NORM(range)` | Euclidean norm √Σx² | numbers/range | `=L2NORM(C2:C10)` |
| `LAG(series, k)` | Shift series down by k rows (previous value) | series, k | `=LAG(B2:B100, 1)` |
| `LEAD(series, k)` | Shift series up by k rows (next value) | series, k | `=LEAD(B2:B100, 1)` |
| `LOG1P(x)` | Natural log of (1 + x) with high precision for small x | x | `=LOG1P(0.05)` |
| `LOG2(x)` | Base-2 logarithm of x | x | `=LOG2(8)` |
| `LOGISTIC(x)` | Logistic sigmoid 1 ⁄ (1 + e^(−x)) | x | `=LOGISTIC(0.3)` |
| `MACD_VAL(series, fast, slow, signal)` | MACD oscillator value: (fast EMA − slow EMA) minus signal EMA | series, fast, slow, signal | `=MACD_VAL(B2:B200, 12, 26, 9)` |
| `MAD(range)` | Mean absolute deviation from mean | numbers/range | `=MAD(C2:C20)` |
| `MADPERCENT(range)` | Median absolute deviation divided by median (robust dispersion %) | numbers/range | `=MADPERCENT(C2:C20)` |
| `MAE(y_true, y_pred)` | Mean absolute error between two sequences | actual_range, predicted_range | `=MAE(C2:C20, D2:D20)` |
| `MANHATTAN(a, b)` | Manhattan distance Σ\|aᵢ − bᵢ\| | vector_a, vector_b | `=MANHATTAN(A1:E1, A2:E2)` |
| `MAPE(y_true, y_pred)` | Mean absolute percentage error (%) | actual_range, predicted_range | `=MAPE(C2:C20, D2:D20)` |
| `MAXIFS(range, crit_range1, crit1, …)` | Maximum in range meeting all criteria | range, criteria_range, criteria | `=MAXIFS(A2:A100, B2:B100, ">0")` |
| `MEDIAN_ABS_DEV(range)` | Median absolute deviation from median | numbers/range | `=MEDIAN_ABS_DEV(C2:C20)` |
| `MEDIAN_HIGH(range)` | Higher median when data count is even | numbers/range | `=MEDIAN_HIGH(C2:C11)` |
| `MEDIAN_LOW(range)` | Lower median when data count is even | numbers/range | `=MEDIAN_LOW(C2:C11)` |
| `MINIFS(range, crit_range1, crit1, …)` | Minimum in range meeting all criteria | range, criteria_range, criteria | `=MINIFS(A2:A100, B2:B100, "<0")` |
| `MINMAX_SCALE(x, min, max)` | Linear min-max normalization of x to [0, 1] | x, min, max | `=MINMAX_SCALE(75, 0, 100)` |
| `MODEMULT(range)` | List of all modes (values with highest frequency) | numbers/range | `=MODEMULT(C2:C20)` |
| `MODE_SCI(range)` | Single mode (most frequent value) | numbers/range | `=MODE_SCI(C2:C20)` |
| `MOM(series, period)` | Momentum: current value − value `period` steps ago | series, period | `=MOM(B2:B200, 5)` |
| `NORMALIZE(range)` | Linearly scale values to [0, 1] | numbers/range | `=NORMALIZE(C2:C20)` |
| `NPM(revenue, net_income)` | Net profit margin = net income ⁄ revenue | revenue, net_income | `=NPM(20000, 3500)` |
| `OBV(close, volume)` | On-Balance Volume cumulative indicator | close_series, volume_series | `=OBV(C2:C200, D2:D200)` |
| `PERCENTRANK(range, x)` | Percentile rank of x within range (0–1) | numbers/range, x | `=PERCENTRANK(C2:C100, 42)` |
| `PERMUT(n, k)` | Number of k-permutations of n items (nPk) | n, k | `=PERMUT(10, 3)` |
| `QUARTILE(range, q)` | Quartile 0–4 of data (0=min, 2=median, 4=max) | numbers/range, q | `=QUARTILE(C2:C100, 1)` |
| `RANGE(range)` | Range of data = max − min | numbers/range | `=RANGE(C2:C100)` |
| `REGEX(text, pattern, group)` | Returns first regex match (group index) or empty string | text, pattern, group | `=REGEX("ab123", "\\d+", 0)` |
| `RESID(y, x)` | Regression residuals y − (intercept + slope·x) | y_range, x_range | `=RESID(C2:C20, B2:B20)` |
| `REVERSE(range)` | Returns the values of range in reverse order | range | `=REVERSE(B2:B10)` |
| `RMSE(y_true, y_pred)` | Root mean square error between sequences | actual_range, predicted_range | `=RMSE(C2:C20, D2:D20)` |
| `ROC_VAL(series, period)` | Rate of change percentage over the period | series, period | `=ROC_VAL(B2:B200, 12)` |
| `ROUNDBANKER(x)` | Rounds x using Banker's rounding (half-to-even) | x | `=ROUNDBANKER(2.5)` |
| `RSI_VAL(series, period)` | Relative Strength Index oscillator value (0–100) | series, period | `=RSI_VAL(B2:B200, 14)` |
| `RSQUARED(y_range, x_range)` | Coefficient of determination r² of linear regression | y_range, x_range | `=RSQUARED(C2:C20, B2:B20)` |
| `SECH(x)` | Hyperbolic secant 1 / cosh(x) | x | `=SECH(1.5)` |
| `SEMI_VARIANCE(range)` | Downside variance: variance of values below mean | numbers/range | `=SEMI_VARIANCE(C2:C20)` |
| `SIGMOID(x)` | Logistic sigmoid 1 ⁄ (1 + e^(−x)) | x | `=SIGMOID(0.7)` |
| `SIGNIF(x, digits)` | Rounds x to the given number of significant digits | x, digits | `=SIGNIF(1234.56, 3)` |
| `SIMPLE_FORECAST(last, growth)` | Simple growth forecast: last × (1 + growth) | last_value, growth | `=SIMPLE_FORECAST(100, 0.05)` |
| `SINC(x)` | Normalised sinc = sin(πx)/(πx) | x | `=SINC(0.5)` |
| `SLN(cost, salvage, life)` | Straight-line depreciation per period | cost, salvage, life | `=SLN(10000, 1000, 5)` |
| `SLOPE(x_range, y_range)` | Linear regression slope of y on x | x_range, y_range | `=SLOPE(B2:B20, C2:C20)` |
| `SLOPELINE(y_range, x_range)` | Slope of best-fit regression line (alias of SLOPE) | y_range, x_range | `=SLOPELINE(C2:C20, B2:B20)` |
| `SMA(series, window)` | Simple moving average over specified window | series, window | `=SMA(B2:B100, 10)` |
| `SMAPE(y_true, y_pred)` | Symmetric mean absolute percentage error (%) | actual_range, predicted_range | `=SMAPE(C2:C20, D2:D20)` |
| `SOFTMAX(v1, v2, …)` | Softmax transform to probabilities summing to 1 | values… | `=SOFTMAX(1, 2, 3)` |
| `SORTASC(range)` | Values of range sorted in ascending order | range | `=SORTASC(C2:C10)` |
| `SORTDESC(range)` | Values of range sorted in descending order | range | `=SORTDESC(C2:C10)` |
| `SQRTPI(n)` | Square root of n × π | n | `=SQRTPI(5)` |
| `SQUARE(x)` | Square of x (x²) | x | `=SQUARE(9)` |
| `STANDARDIZE(range)` | Rescale to mean 0 and std-dev 1 | numbers/range | `=STANDARDIZE(C2:C20)` |
| `STDERR(values…)` | Standard error of the mean of values | values… | `=STDERR(C2:C20)` |
| `SUMPRODUCT(range1, range2, …)` | Sum of element-wise products (e.g. weighted sums) | range1, range2, … | `=SUMPRODUCT(A1:A3, B1:B3)` |
| `SYD(cost, salvage, life, period)` | Depreciation via Sum-of-Years'-Digits method | cost, salvage, life, period | `=SYD(10000, 1000, 5, 2)` |
| `TRANSPOSE(range)` | Flips rows and columns of the range | range | `=TRANSPOSE(A1:C2)` |
| `TTEST(sample1, sample2)` | Welch two-sample t-statistic | sample1_range, sample2_range | `=TTEST(B2:B20, C2:C20)` |
| `UNIQUECNT(range)` | Count of unique distinct values | range | `=UNIQUECNT(A1:A100)` |
| `UUID()` | Generate random UUID-4 string | — | `=UUID()` |
| `VWAP_VAL(price, volume)` | Volume-Weighted Average Price over period | price_series, volume_series | `=VWAP_VAL(C2:C200, D2:D200)` |
| `WEIGHTED_MEAN(values, weights)` | Weighted arithmetic mean Σvᵢwᵢ ⁄ Σwᵢ | values_range, weights_range | `=WEIGHTED_MEAN(C2:C10, D2:D10)` |
| `WORKINGCAP(current_assets, current_liabilities)` | Working capital = current assets − current liabilities | current_assets, current_liabilities | `=WORKINGCAP(15000, 8000)` |
| `ZSCORE_SCALE(x, mean, sd)` | Z-score standardisation (x − mean) ⁄ sd | x, mean, sd | `=ZSCORE_SCALE(85, 70, 10)` |

## Rolling

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `EMAVG` | Exponential moving average | series, window | `=EMAVG(B2:B100, 10)` |
| `MOVAVG` | Simple moving average | series, window | `=MOVAVG(B2:B100, 10)` |
| `MOVING_AVG` | Simple moving average | series, window | `=MOVING_AVG(B2:B100, 10)` |
| `PCTCHANGE` | Percentage change | series | `=PCTCHANGE(B2:B100)` |
| `ROLLINGMAX` | Rolling maximum | series, window | `=ROLLINGMAX(B2:B100, 14)` |
| `ROLLINGMEAN` | Rolling mean | series, window | `=ROLLINGMEAN(B2:B100, 14)` |
| `ROLLINGMIN` | Rolling minimum | series, window | `=ROLLINGMIN(B2:B100, 14)` |
| `ROLLINGSUM` | Rolling window sum | series, window | `=ROLLINGSUM(B2:B100, 14)` |

## Statistical

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `AVERAGEIF` | Average with condition | range, condition | `=AVERAGEIF(A2:A20, ">0")` |
| `AVERAGEIFS` | Average with multiple conditions | range, crit_range1, crit1, … | `=AVERAGEIFS(A2:A20, B2:B20, ">0", C2:C20, "<5")` |
| `COUNTIF(range, crit)` | Count with condition | range, crit | `=COUNTIF(A2:A100, ">0")` |
| `COUNTIFS(range1, crit_range1, crit1, …)` | Count with multiple conditions | range1, crit_range1, crit1, … | `=COUNTIFS(A2:A100, ">0", B2:B100, "<5")` |
| `KURT(range)` | Kurtosis (tailedness) of distribution | numbers/range | `=KURT(C2:C100)` |
| `LARGE(range, k)` | k-th largest value | range, k | `=LARGE(C2:C100, 3)` |
| `MEDIAN(range)` | Median value | numbers/range | `=MEDIAN(C2:C100)` |
| `MODE(range)` | Most frequent value | numbers/range | `=MODE(C2:C100)` |
| `PERCENTILE(range, k)` | k-th percentile | range, k | `=PERCENTILE(C2:C100, 0.9)` |
| `PERCENTILE_EXC(range, k)` | Exclusive percentile | range, k | `=PERCENTILE_EXC(C2:C100, 0.9)` |
| `PERCENTILE_INC(range, k)` | Inclusive percentile | range, k | `=PERCENTILE_INC(C2:C100, 0.9)` |
| `RAND()` | Random 0–1 | — | `=RAND()` |
| `RANDBETWEEN(low, high)` | Random integer in range | low, high | `=RANDBETWEEN(1, 100)` |
| `SKEW(range)` | Skewness (asymmetry) of distribution | numbers/range | `=SKEW(C2:C100)` |
| `SMALL(range, k)` | k-th smallest value | range, k | `=SMALL(C2:C100, 2)` |
| `STDEV(range)` | Sample std-dev | numbers/range | `=STDEV(C2:C100)` |
| `STDEVP(range)` | Population std-dev | numbers/range | `=STDEVP(C2:C100)` |
| `SUMIF(range, crit)` | Sum with condition | range, crit | `=SUMIF(A2:A100, ">0")` |
| `SUMIFS(sum_range, crit_range1, crit1, …)` | Sum with multiple conditions | sum_range, crit_range1, crit1, … | `=SUMIFS(A2:A100, B2:B100, ">0", C2:C100, "<5")` |
| `VAR(range)` | Sample variance | numbers/range | `=VAR(C2:C100)` |
| `VARP(range)` | Population variance | numbers/range | `=VARP(C2:C100)` |
| `ZSCORE(x, mean, std)` | Standard score (x − mean)/std | x, mean, std | `=ZSCORE(82, 70, 10)` |

## Text

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `CONCAT(text1, text2, …)` | Concatenate strings | text1, text2, … | `=CONCAT("Hello", " ", "World")` |
| `FIND(substr, text, start)` | Position of substring | substr, text, start? | `=FIND("cat", "concatenate", 1)` |
| `LEFT(text, n)` | Leftmost characters | text, n | `=LEFT("Spreadsheet", 6)` |
| `LEN(text)` | Text length | text | `=LEN("OpenAI")` |
| `LOWER(text)` | Lowercase text | text | `=LOWER("HeLLo")` |
| `MID(text, start, len)` | Substring from position | text, start, len | `=MID("Spreadsheet", 3, 5)` |
| `PROPER(text)` | Capitalise each word | text | `=PROPER("hello world")` |
| `REPLACE(text, start, len, new_text)` | Replace part of text | text, start, len, replacement | `=REPLACE("12345", 2, 3, "abc")` |
| `REPT(text, n)` | Repeat text n times | text, n | `=REPT("*", 5)` |
| `RIGHT(text, n)` | Rightmost characters | text, n | `=RIGHT("Spreadsheet", 4)` |
| `SPLIT(text, delim)` | Split text by delimiter | text, delimiter | `=SPLIT("a,b,c", ",")` |
| `SUBSTITUTE(text, old, new, instance)` | Replace occurrences of text | text, old, new, instance? | `=SUBSTITUTE("banana", "a", "o", 2)` |
| `TEXTJOIN(delim, ignore_empty, text1, …)` | Join text with delimiter | delim, ignore_empty, texts… | `=TEXTJOIN("-", TRUE, "a", "b", "c")` |
| `TRIM(text)` | Remove leading/trailing spaces | text | `=TRIM("  hello  ")` |
| `UPPER(text)` | Uppercase text | text | `=UPPER("hello")` |
| `VALUE(text)` | Text to number | text | `=VALUE("123.45")` |
| `SEARCH(find_text, within_text, start?)` | Case-insensitive position (1-based), 0 if not found | find_text, within_text, start? | `=SEARCH("cat", "Concatenate", 1)` |
| `TEXT(value, format)` | Format number/date as text | value, format | `=TEXT(42370, "yyyy-mm-dd")` |

## Trigonometry

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `ACOS(x)` | Arccosine | x | `=ACOS(0.5)` |
| `ASIN(x)` | Arcsine | x | `=ASIN(0.5)` |
| `ATAN(x)` | Arctangent | x | `=ATAN(1)` |
| `ATAN2(y, x)` | Arctangent of y/x | y, x | `=ATAN2(1, 1)` |
| `COS(theta)` | Cosine of angle (rad) | angle_rad | `=COS(PI()/3)` |
| `COSH(x)` | Hyperbolic cosine | x | `=COSH(1)` |
| `COT(theta)` | Cotangent | angle_rad | `=COT(PI()/4)` |
| `CSC(theta)` | Cosecant | angle_rad | `=CSC(PI()/6)` |
| `DEGREES(rad)` | Radians→degrees | angle_rad | `=DEGREES(PI())` |
| `RADIANS(deg)` | Degrees→radians | angle_deg | `=RADIANS(180)` |
| `SEC(theta)` | Secant | angle_rad | `=SEC(PI()/4)` |
| `SIN(theta)` | Sine of angle (rad) | angle_rad | `=SIN(PI()/6)` |
| `SINH(x)` | Hyperbolic sine | x | `=SINH(1)` |
| `TAN(theta)` | Tangent (rad) | angle_rad | `=TAN(PI()/4)` |
| `TANH(x)` | Hyperbolic tangent | x | `=TANH(1)` |

## Information

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `ISNUMBER(x)` | Returns TRUE if x is numeric | value | `=ISNUMBER(123)` |
| `ISTEXT(x)` | TRUE if x is text | value | `=ISTEXT("abc")` |
| `ISBLANK(x)` | TRUE if x is blank/empty | value | `=ISBLANK(A1)` |
| `ISEVEN(n)` | TRUE if integer n is even | n | `=ISEVEN(4)` |
| `ISODD(n)` | TRUE if integer n is odd | n | `=ISODD(5)` |

## Lookup & Reference

| Function | Description | Input | Example |
|----------|-------------|-------|---------|
| `INDEX(range, row, col)` | Value at given row & column in range | range, row_num, col_num | `=INDEX(A1:C3, 2, 1)` |
| `MATCH(value, vector)` | Position of value in vector (1-based) | value, vector | `=MATCH("apple", A1:A10)` |
| `VLOOKUP(value, table, col)` | Lookup value in first column, return from `col` | value, table_range, col_index | `=VLOOKUP("ID42", A1:C100, 3)` |
| `HLOOKUP(value, table, row)` | Lookup value in first row, return from `row` | value, table_range, row_index | `=HLOOKUP("Q1", A1:D5, 2)` |
| `XLOOKUP(value, lookup_vector, return_vector)` | Flexible lookup with separate vectors | value, lookup_vector, return_vector | `=XLOOKUP("Bob", A1:A10, B1:B10)` |