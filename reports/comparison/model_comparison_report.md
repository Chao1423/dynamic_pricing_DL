# Model Comparison Report

## 1. Overall Model Performance

| Model          |   MAE (Original) |   RMSE (Original) |       R² |   MAPE (%) |
|:---------------|-----------------:|------------------:|---------:|-----------:|
| mlp_embeddings |          2507.39 |           4614.67 | 0.960473 |    14.6122 |
| tabtransformer |          3722.6  |           6425.18 | 0.953754 |    18.5735 |
| linear         |          4920.61 |           8358.46 | 0.894959 |    29.712  |

**Best Model (by MAE):** mlp_embeddings (MAE = 2507.39)

## 2. Grouped Error Analysis

### 2.1 By Airline

#### linear

**Top 5 Airlines with Highest MAE:**

|   airline |     mae |   count |
|----------:|--------:|--------:|
|         6 | 6946.87 |   25267 |
|         2 | 5371.25 |   16076 |
|         5 | 1996.12 |    1771 |
|         1 | 1819.97 |    3250 |
|         4 | 1748.04 |    8652 |

**Top 5 Airlines with Lowest MAE:**

|   airline |     mae |   count |
|----------:|--------:|--------:|
|         2 | 5371.25 |   16076 |
|         5 | 1996.12 |    1771 |
|         1 | 1819.97 |    3250 |
|         4 | 1748.04 |    8652 |
|         3 | 1512.15 |    4617 |

#### mlp_embeddings

**Top 5 Airlines with Highest MAE:**

|   airline |     mae |   count |
|----------:|--------:|--------:|
|         6 | 3510.66 |   25267 |
|         2 | 2524.16 |   16076 |
|         5 | 1257.01 |    1771 |
|         4 | 1170.42 |    8652 |
|         3 | 1014.42 |    4617 |

**Top 5 Airlines with Lowest MAE:**

|   airline |      mae |   count |
|----------:|---------:|--------:|
|         2 | 2524.16  |   16076 |
|         5 | 1257.01  |    1771 |
|         4 | 1170.42  |    8652 |
|         3 | 1014.42  |    4617 |
|         1 |  986.046 |    3250 |

#### tabtransformer

**Top 5 Airlines with Highest MAE:**

|   airline |     mae |   count |
|----------:|--------:|--------:|
|         6 | 5584.01 |   25267 |
|         2 | 3729.32 |   16076 |
|         5 | 1393.27 |    1771 |
|         4 | 1189.41 |    8652 |
|         3 | 1058.02 |    4617 |

**Top 5 Airlines with Lowest MAE:**

|   airline |     mae |   count |
|----------:|--------:|--------:|
|         2 | 3729.32 |   16076 |
|         5 | 1393.27 |    1771 |
|         4 | 1189.41 |    8652 |
|         3 | 1058.02 |    4617 |
|         1 | 1016.33 |    3250 |

### 2.2 By Route

#### linear

**Top 10 Routes with Highest MAE:**

| route   |     mae |   count |
|:--------|--------:|--------:|
| 2 -> 1  | 6644.18 |    1308 |
| 4 -> 1  | 6308.48 |    1528 |
| 1 -> 2  | 6236.75 |    1281 |
| 1 -> 4  | 6101.1  |    1722 |
| 4 -> 2  | 6050.76 |    1243 |
| 2 -> 4  | 5801.15 |    1236 |
| 3 -> 6  | 5391.18 |    3094 |
| 5 -> 2  | 5386.44 |    1290 |
| 6 -> 1  | 5358.08 |    2542 |
| 6 -> 3  | 5273.33 |    2943 |

#### mlp_embeddings

**Top 10 Routes with Highest MAE:**

| route   |     mae |   count |
|:--------|--------:|--------:|
| 2 -> 1  | 3682.44 |    1308 |
| 4 -> 1  | 3466.04 |    1528 |
| 3 -> 6  | 3269.59 |    3094 |
| 1 -> 2  | 3232.81 |    1281 |
| 4 -> 2  | 3125.48 |    1243 |
| 2 -> 4  | 3063.83 |    1236 |
| 6 -> 3  | 2981.13 |    2943 |
| 2 -> 6  | 2950.26 |    1801 |
| 1 -> 5  | 2904.96 |    1950 |
| 1 -> 4  | 2721.08 |    1722 |

#### tabtransformer

**Top 10 Routes with Highest MAE:**

| route   |     mae |   count |
|:--------|--------:|--------:|
| 2 -> 1  | 4983.28 |    1308 |
| 5 -> 2  | 4665.21 |    1290 |
| 5 -> 4  | 4528.64 |    1587 |
| 2 -> 4  | 4472.12 |    1236 |
| 2 -> 5  | 4339.3  |    1344 |
| 5 -> 6  | 4325.9  |    2204 |
| 4 -> 2  | 4304.32 |    1243 |
| 4 -> 1  | 4243.62 |    1528 |
| 1 -> 5  | 4191.01 |    1950 |
| 6 -> 1  | 4184.92 |    2542 |

### 2.3 By Number of Stops

#### linear

|   stops |     mae |   count |
|--------:|--------:|--------:|
|       2 | 5491.54 |    2457 |
|       1 | 5202.15 |   50059 |
|       3 | 2743.21 |    7117 |

#### mlp_embeddings

|   stops |     mae |   count |
|--------:|--------:|--------:|
|       1 | 2732.54 |   50059 |
|       2 | 2013.17 |    2457 |
|       3 | 1094.36 |    7117 |

#### tabtransformer

|   stops |     mae |   count |
|--------:|--------:|--------:|
|       1 | 4053.54 |   50059 |
|       2 | 2763.72 |    2457 |
|       3 | 1725.91 |    7117 |

## 3. TabTransformer Improvements

### 3.1 Top Improvements by Airline

TabTransformer shows largest improvements (vs MLP) on:

- **4**: -1.62% improvement
- **1**: -3.07% improvement
- **3**: -4.30% improvement
- **5**: -10.84% improvement
- **2**: -47.75% improvement
- **6**: -59.06% improvement

## 4. High Error Categories

Categories where all models show high error:

**Airlines with consistently high error:**

- 6.0: MAE = 5584.01 (n=25267.0)
- 2.0: MAE = 3729.32 (n=16076.0)

## 5. Summary

### Key Findings:

1. **Best Overall Model**: mlp_embeddings
2. TabTransformer shows improvements in capturing categorical interactions
3. Deep learning models (MLP, TabTransformer) outperform linear baseline
4. Error varies significantly across airlines and routes

