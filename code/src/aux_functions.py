import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

SELECTED_VARS = ['accel_z_10_ewm',
 'feat_time',
 'feat_knn_class_0_5',
 'feat_knn_class_0_13',
 'accel_z_10_min',
 'accel_y_5_ewm',
 'feat_cat_knn_class_0_3',
 'feat_accelcos_yz_5_midpoint',
 'gyro_x_200_ewm',
 'feat_knn_class_0_10',
 'gyro_z_20_closetoclose',
 'accel_z_5_max',
 'accel_y_20_dema',
 'feat_speeddiff_xz_5_trix',
 'gyro_z_200_ewm',
 'accel_x_jerk_50_200_ewm',
 'gyro_y_150_ewm',
 'ventile_cat_time',
 'accel_y_5_min',
 'sin_position_z_5_rogersatchell',
 'feat_acceldiff_xy_20_dema',
 'feat_acceldiff_yz_5_ewm',
 'feat_acceldiff_xy_20_t3',
 'feat_cat_knn_class_0_20',
 'accel_z_5_ewm',
 'accel_y_jerk_20_200_ewm',
 'feat_speeddiff_yz_100_coef',
 'accel_y_5_mean',
 'accel_y_5_midpoint',
 'feat_acceldiff_xz_200_t3',
 'feat_acceldiff_xy_200_rsi',
 'feat_speeddiff_xz_200_coef',
 'accel_y_20_t3',
 'feat_knn_class_0_2',
 'accel_y_10_min',
 'speed_x_10_closetoclose',
 'feat_acceldiff_yz_20_t3',
 'feat_cat_knn_class_0_0',
 'accel_z_5_min',
 'speed_y_200_coef',
 'sin_position_y_100_coef',
 'position_z_200_ewm',
 'feat_speeddiff_xz_10_cmo',
 'feat_accel_normxy_50_ewm',
 'accel_y_200_rsi',
 'accel_y_5_max',
 'accel_z_5_midpoint',
 'gyro_x_200_mean',
 'feat_acceldiff_yz_5_min',
 'accel_x_jerk_50_5_std',
 'accel_y_10_dema',
 'position_x_200_parkinson',
 'feat_speeddiff_xy_10_lin',
 'feat_acceldiff_yz_200_t3',
 'feat_acceldiff_yz_10_dema',
 'accel_x_jerk_100_kurt',
 'feat_accelsin_xz_20_ewm',
 'feat_abs_gyroy_20_ewm',
 'sin_position_y_100_max',
 'accel_z_10_t3',
 'accel_z_20_t3',
 'speed_z_100_min',
 'sin_position_z_20_rsi',
 'accel_y_100_min',
 'accel_z_20_ewm',
 'gyro_y_10_closetoclose',
 'feat_accelangle_xz_10_trima',
 'sin_position_z_200_rsi',
 'feat_cat_knn_class_0_18',
 'gyro_z_200_midpoint',
 'gyro_z_200_mean',
 'gyro_z_100_mean',
 'speed_z_10_cmo',
 'accel_z_200_max',
 'feat_abs_gyroz_10_midpoint',
 'accel_y_5_germanklass',
 'accel_y_10_max',
 'feat_accel_norm_10_min',
 'speed_z_100_coef',
 'feat_acceldiff_xz_200_min',
 'feat_acceldiff_xy_10_midpoint',
 'feat_acceldiff_yz_20_dema',
 'feat_speeddiff_xz_200_rsi',
 'position_x_200_ewm',
 'accel_xz_corr_20',
 'feat_speeddiff_xy_5_trix',
 'accel_z_50_t3',
 'gyro_y_100_ewm',
 'gyro_z_100_t3',
 'speed_y_5_change',
 'gyro_x_100_t3',
 'feat_accel_normxy_50_min',
 'gyro_x_150_ewm',
 'accel_x_5_rogersatchell',
 'feat_speeddiff_xy_5_cmo',
 'feat_speeddiff_xy_200_trix',
 'gyro_x_50_trima',
 'gyro_z_100_dema',
 'speed_z_200_rsi',
 'accel_x_10_max',
 'accel_y_150_min',
 'feat_cat_knn_class_0_4',
 'feat_acceldiff_yz_10_min',
 'feat_gyro_norm_10_dema',
 'speed_x_100_t3',
 'accel_y_20_std',
 'feat_accelangle_xy_20_dema',
 'feat_abs_gyrox_100_trix',
 'accel_x_100_t3',
 'feat_acceldiff_yz_200_min',
 'gyro_y_50_trima',
 'speed_x_100_min',
 'gyro_y_20_trima',
 'accel_z_100_t3',
 'feat_acceldiff_xz_5_min',
 'speed_x_100_coef',
 'gyro_x_50_rsi',
 'feat_accel_normyz_100_trix',
 'speed_y_5_cmo',
 'feat_accelcos_xz_5_rogersatchell',
 'cos_position_z_200_rsi',
 'feat_acceldiff_yz_10_ewm',
 'accel_y_10_ewm',
 'sin_position_z_50_min',
 'feat_acceldiff_xz_100_t3',
 'feat_acceldiff_xz_10_max',
 'feat_accelsin_xy_20_ewm',
 'gyro_y_20_rsi',
 'feat_accel_normxz_20_max',
 'speed_z_100_max',
 'feat_speeddiff_yz_200_rsi',
 'position_x',
 'accel_z_150_t3',
 'feat_speeddiff_yz_50_dema',
 'position_y_50_trix',
 'accel_z_20_min',
 'accel_z_200_midpoint',
 'speed_x_5_rogersatchell',
 'accel_yz_corr_10',
 'sin_position_x_150_rsi',
 'feat_accel_normxy_20_std',
 'feat_acceldiff_yz_100_t3',
 'feat_accelcos_xy_100_trix',
 'speed_y_100_max',
 'feat_acceldiff_xy_10_min',
 'accel_x_jerk_150_kurt',
 'feat_accel_normyz_20_trix',
 'accel_x_jerk_100_closetoclose',
 'gyro_z_150_mean',
 'feat_abs_gyrox_100_std',
 'feat_speeddiff_yz_100_t3',
 'gyro_x_100_min',
 'feat_accel_normxy_20_min',
 'accel_z_50_max',
 'feat_speeddiff_xy_5_parkinson',
 'feat_accel_normxz_10_max',
 'gyro_x_200_mad',
 'gyro_x_150_mean',
 'feat_gyro_normyz_100_std',
 'speed_x_5_trix',
 'feat_acceldiff_xy_5_min',
 'feat_speeddiff_xz_10_std',
 'accel_x_200_max',
 'gyro_z_200_closetoclose',
 'accel_z_100_rsi',
 'position_y_100_parkinson',
 'feat_acceldiff_yz_100_max',
 'accel_z_200_t3',
 'feat_accelangle_xz_10_max',
 'accel_z_5_mean',
 'feat_speeddiff_xz_100_min',
 'feat_accelangle_yz_5_min',
 'speed_x_100_trix',
 'feat_acceldiff_xz_10_midpoint',
 'feat_acceldiff_xz_10_min',
 'feat_acceldiff_xy_200_min',
 'position_z_150_ewm',
 'sin_position_z_100_midpoint',
 'gyro_x_100_ewm',
 'feat_accelsin_xz_200_ewm',
 'sin_position_y_100_parkinson',
 'speed_y_150_coef',
 'feat_acceldiff_xy_10_max',
 'position_y_200_trix',
 'feat_speeddiff_xy_150_trix',
 'gyro_y_100_kurt',
 'speed_z_5_cmo',
 'gyro_y_100_min',
 'accel_z_100_max',
 'feat_acceldiff_xy_5_midpoint',
 'cos_position_y_100_dema',
 'feat_accelangle_xy_10_ewm',
 'feat_speeddiff_yz_20_rsi',
 'feat_speeddiff_xz_150_coef',
 'accel_z_10_max',
 'feat_abs_gyrox_100_mad',
 'feat_abs_gyroz_100_trix',
 'feat_acceldiff_yz_20_max',
 'feat_acceldiff_yz_20_min',
 'gyro_x_100_mean',
 'speed_z_20_cmo',
 'accel_y_jerk_200_mad',
 'gyro_y_100_mean',
 'position_x_100_parkinson',
 'accel_y_jerk_20_mad',
 'gyro_y_20_closetoclose',
 'accel_y_200_midpoint',
 'gyro_x_100_max',
 'accel_xz_corr_100',
 'accel_y_50_dema',
 'gyro_z_50_mean',
 'feat_accelangle_xy_10_min',
 'gyro_x_150_mad',
 'speed_z_5_change',
 'position_change_z_50_100_max',
 'speed_y_50_max',
 'gyro_z_150_ewm',
 'accel_y_100_max',
 'accel_x_jerk_50_200_std',
 'feat_speeddiff_xy_100_trix',
 'feat_knn_class_0_12',
 'cos_position_y_100_t3',
 'speed_x_50_trix',
 'sin_position_z_200_min',
 'sin_position_x_200_rsi',
 'accel_z_50_ewm',
 'feat_speeddiff_xz_100_dema',
 'feat_accel_norm_100_trima',
 'feat_abs_accelx_100_trix',
 'feat_accelangle_xz_100_min',
 'feat_accel_norm_100_min',
 'cos_position_z_200_trix',
 'gyro_y_150_mean',
 'accel_z_50_min',
 'speed_z_5_trix',
 'feat_acceldiff_xy_50_ewm',
 'accel_z_100_ewm',
 'feat_accelangle_xy_200_dema',
 'gyro_z_50_closetoclose',
 'speed_z_20_rsi',
 'feat_accelcos_yz_20_dema',
 'feat_accelsin_yz_100_dema',
 'accel_y_50_min',
 'feat_acceldiff_yz_150_t3',
 'feat_acceldiff_yz_150_min',
 'gyro_z_100_midpoint',
 'accel_y_jerk_20_100_min',
 'feat_accelsin_xz_150_ewm',
 'feat_accelcos_yz_100_ewm',
 'feat_speeddiff_yz_10_rsi']


def ventile_time(x):
    cuts = [178, 356, 535, 713, 891, 1070, 1249, 1428, 1607, 1787,
            1968, 2150, 2335, 2524, 2718, 2920, 3130, 3363, 4114]
    for i in range(len(cuts)):
        if x < cuts[i]:
            return i
    return len(cuts)

def TRIX_T3_DEMA(close, period):
    # Triangular and triple 
    ema1 = close.ewm(min_periods=period, span=period, adjust=False).mean()
    ema2 = ema1.ewm(min_periods=period, span=period, adjust=False).mean()
    ema3 = ema2.ewm(min_periods=period, span=period, adjust=False).mean()
    t3 = 3 * ema1 - 3 * ema2 + ema3
    trix = 100 * (ema3 - ema3.shift()) / (ema3 + 0.0001)
    dema = 2 * ema1 - ema2
    return trix, t3, dema

def ADD_LIN(close, period):
    return close.rolling(period).apply(lambda x: np.polyfit(range(period), x, 1)[0])

def ADD_COEF(close, period):
    return close.rolling(period).apply(lambda x: np.polyfit(range(period), x, 1)[1])

def CMO(close, period):
    # Chande Momentum Oscillator
    shifted_close = close.shift()
    up = (close - shifted_close).clip(lower=0)
    down = (shifted_close - close).clip(lower=0)
    ups = up.rolling(period).sum()
    downs = down.rolling(period).sum()
    cmo = 100 * (ups - downs) / (ups + downs + 0.0001)
    return cmo


def RSI(close, period):
    # Relative strength index
    close_delta = close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


def MIDPOINT(close, period):
    return (close.rolling(period).max() + close.rolling(period).min()) / 2


def TRIMA(close, period):
    # Triangular moving average
    nm = round((period + 1) / 2)
    SMAm = close.rolling(nm).mean()
    trima = SMAm.rolling(nm).mean()
    return trima


def CloseToClose_estimator(close, period):
    log_return = (close / close.shift(1)).apply(np.log)
    result = log_return.rolling(window=period, center=False).std()
    return result


def Parkinson_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    rs = (1.0 / (4.0 * np.log(2.0))) * ((High / Low).apply(np.log)) ** 2.0
    result = rs.rolling(window=10).mean()
    return result


def GarmanKlass_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    Open = close.shift(period)

    log_hl = (High / Low).apply(np.log)
    log_co = (close / Open).apply(np.log)
    rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    result = rs.rolling(window=10).mean()
    return result


def RogerSatchell_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    Open = close.shift(period)
    log_ho = (High / Open).apply(np.log)
    log_lo = (Low / Open).apply(np.log)
    log_co = (close / Open).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    result = rs.rolling(window=10).mean()
    return result


def strided_app(a, L, S):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mad_numpy(a, W):
    if W > len(a):
        return [np.nan for i in range(len(a))]
    a2D = strided_app(a, W, 1)
    mad = np.absolute(a2D - moving_average(a, W)[:, None]).mean(1)
    mad = np.concatenate([[np.nan for i in range(W - 1)], mad])
    return mad


def extract_position_features(df):
    """
    Given a dataframe containing an only session, extracts position and sin/cos of position degrees
    as the cumulative sum of gyro changes from the beginning of the time series history
    """

    # Define parameters of position normalization
    modulo = 360
    units_per_sec = 26
    fraction_degrees = 4.48

    # Retrieve position coordinates from movement gyro data
    df['position_x'] = (df['gyro_x'] / units_per_sec / fraction_degrees).cumsum() % modulo
    df['position_y'] = (df['gyro_y'] / units_per_sec / fraction_degrees).cumsum() % modulo
    df['position_z'] = (df['gyro_z'] / units_per_sec / fraction_degrees).cumsum() % modulo

    # Extract sin of the position coordinates
    df['sin_position_x'] = np.sin(df['position_x'] / 360 * 2 * np.pi)
    df['sin_position_y'] = np.sin(df['position_y'] / 360 * 2 * np.pi)
    df['sin_position_z'] = np.sin(df['position_z'] / 360 * 2 * np.pi)

    # Extract cos of the position coordinates
    df['cos_position_x'] = np.cos(df['position_x'] / 360 * 2 * np.pi)
    df['cos_position_y'] = np.cos(df['position_y'] / 360 * 2 * np.pi)
    df['cos_position_z'] = np.cos(df['position_z'] / 360 * 2 * np.pi)

    # Extract speed features
    df['speed_x'] = (df['accel_x'] / 0.0312319993972778 - 15).cumsum() / units_per_sec
    df['speed_y'] = (df['accel_y'] / 0.0312319993972778 + 4).cumsum() / units_per_sec
    df['speed_z'] = (df['accel_z'] / 0.0312319993972778 + 3).cumsum() / units_per_sec

    return df


def extract_row_features(df):

    # Extract time from timestep
    df['feat_time'] = df['timestep'].apply(lambda x: str(x).replace('timestep_', '')).astype(int)
    df['ventile_cat_time'] = df['feat_time'].apply(lambda x: ventile_time(x)).astype(int)

    # Extract norms of the acceleration
    df[f'feat_accel_norm'] = np.sqrt(df[f'accel_x'] ** 2 + df[f'accel_y'] ** 2 + df[f'accel_z'] ** 2)
    df[f'feat_accel_normxy'] = np.sqrt(df[f'accel_x'] ** 2 + df[f'accel_y'] ** 2)
    df[f'feat_accel_normxz'] = np.sqrt(df[f'accel_x'] ** 2 + df[f'accel_z'] ** 2)
    df[f'feat_accel_normyz'] = np.sqrt(df[f'accel_y'] ** 2 + df[f'accel_z'] ** 2)

    # Extract norms of the gyro
    df[f'feat_gyro_norm'] = np.sqrt(df[f'gyro_x'] ** 2 + df[f'gyro_y'] ** 2 + df[f'gyro_z'] ** 2)
    df[f'feat_gyro_normxy'] = np.sqrt(df[f'gyro_x'] ** 2 + df[f'gyro_y'] ** 2)
    df[f'feat_gyro_normxz'] = np.sqrt(df[f'gyro_x'] ** 2 + df[f'gyro_z'] ** 2)
    df[f'feat_gyro_normyz'] = np.sqrt(df[f'gyro_y'] ** 2 + df[f'gyro_z'] ** 2)

    # Extract coordinate differences in acceleration
    df['feat_acceldiff_xy'] = df['accel_x'] - df['accel_y']
    df['feat_acceldiff_xz'] = df['accel_x'] - df['accel_z']
    df['feat_acceldiff_yz'] = df['accel_y'] - df['accel_z']

    # Extract angles of change in acceleration
    df['feat_accelangle_xy'] = np.arctan(df.accel_x / df.accel_y).fillna(0)
    df['feat_accelangle_xz'] = np.arctan(df.accel_x / df.accel_z).fillna(0)
    df['feat_accelangle_yz'] = np.arctan(df.accel_y / df.accel_z).fillna(0)

    # Extract sin and cos of the acceleration angles of change
    df['feat_accelsin_xy'] = np.sin(df['feat_accelangle_xy'])
    df['feat_accelsin_xz'] = np.sin(df['feat_accelangle_xz'])
    df['feat_accelsin_yz'] = np.sin(df['feat_accelangle_yz'])
    df['feat_accelcos_xy'] = np.cos(df['feat_accelangle_xy'])
    df['feat_accelcos_xz'] = np.cos(df['feat_accelangle_xz'])
    df['feat_accelcos_yz'] = np.cos(df['feat_accelangle_yz'])

    # Extract absolute values of the acceleration coordinates
    df['feat_abs_accelx'] = abs(df['accel_x'])
    df['feat_abs_accely'] = abs(df['accel_y'])
    df['feat_abs_accelz'] = abs(df['accel_z'])

    # Extract absolute values of the gyro coordinates
    df['feat_abs_gyrox'] = abs(df['gyro_x'])
    df['feat_abs_gyroy'] = abs(df['gyro_y'])
    df['feat_abs_gyroz'] = abs(df['gyro_z'])

    # Extract coordinate differences in speed
    df['feat_speeddiff_xy'] = df['speed_x'] - df['speed_y']
    df['feat_speeddiff_xz'] = df['speed_x'] - df['speed_z']
    df['feat_speeddiff_yz'] = df['speed_y'] - df['speed_z']

    return df


def extract_historic_features(df):

    df['accel_x_jerk'] = df['accel_x'].diff().fillna(0)
    df['accel_y_jerk'] = df['accel_y'].diff().fillna(0)
    df['accel_z_jerk'] = df['accel_z'].diff().fillna(0)

    df['accel_x_jerk_20'] = (df['accel_x'] - df['accel_x'].rolling(20).mean()).fillna(0)
    df['accel_y_jerk_20'] = (df['accel_y'] - df['accel_y'].rolling(20).mean()).fillna(0)
    df['accel_z_jerk_20'] = (df['accel_z'] - df['accel_z'].rolling(20).mean()).fillna(0)

    df['accel_x_jerk_50'] = (df['accel_x'] - df['accel_x'].rolling(50).mean()).fillna(0)
    df['accel_y_jerk_50'] = (df['accel_y'] - df['accel_y'].rolling(50).mean()).fillna(0)
    df['accel_z_jerk_50'] = (df['accel_z'] - df['accel_z'].rolling(50).mean()).fillna(0)

    # Extract period based features
    periods = [5, 10, 20, 50, 100, 150, 200]

    for period in periods:
        if (f'accel_xy_corr_{period}' in SELECTED_VARS):
            df[f'accel_xy_corr_{period}'] = df['accel_x'].rolling(period).corr(df['accel_y'])
        if (f'accel_xz_corr_{period}' in SELECTED_VARS):
            df[f'accel_xz_corr_{period}'] = df['accel_x'].rolling(period).corr(df['accel_z'])
        if (f'accel_yz_corr_{period}' in SELECTED_VARS):
            df[f'accel_yz_corr_{period}'] = df['accel_y'].rolling(period).corr(df['accel_z'])

        # Extract position changes
        df[f'position_change_x_{period}'] = ((df['sin_position_x'] - df['sin_position_x'].shift(period)) ** 2).fillna(0)
        df[f'position_change_y_{period}'] = ((df['sin_position_y'] - df['sin_position_y'].shift(period)) ** 2).fillna(0)
        df[f'position_change_z_{period}'] = ((df['sin_position_z'] - df['sin_position_z'].shift(period)) ** 2).fillna(0)
        df[f'position_change_{period}'] = (df[f'position_change_x_{period}'] +
                                           df[f'position_change_y_{period}'] +
                                           df[f'position_change_z_{period}'])

    # Extract series - period features
    series = ['accel_x', 'accel_y', 'accel_z',
              'gyro_x', 'gyro_y', 'gyro_z',
              'speed_x', 'speed_y', 'speed_z',
              'feat_accel_norm', 'feat_accel_normxy', 'feat_accel_normxz', 'feat_accel_normyz',
              'feat_gyro_norm', 'feat_gyro_normxy', 'feat_gyro_normxz', 'feat_gyro_normyz',
              'feat_acceldiff_xy', 'feat_acceldiff_xz', 'feat_acceldiff_yz',
              'feat_accelangle_xy', 'feat_accelangle_xz', 'feat_accelangle_yz',
              'feat_accelsin_xy', 'feat_accelsin_xz', 'feat_accelsin_yz',
              'feat_accelcos_xy', 'feat_accelcos_xz', 'feat_accelcos_yz',
              'feat_abs_accelx', 'feat_abs_accely', 'feat_abs_accelz',
              'feat_abs_gyrox', 'feat_abs_gyroy', 'feat_abs_gyroz',
              'position_x', 'position_y', 'position_z',
              'sin_position_x', 'sin_position_y', 'sin_position_z',
              'cos_position_x', 'cos_position_y', 'cos_position_z',
              'accel_x_jerk', 'accel_y_jerk', 'accel_z_jerk',
              'accel_x_jerk_20', 'accel_y_jerk_20', 'accel_z_jerk_20',
              'accel_x_jerk_50', 'accel_y_jerk_50', 'accel_z_jerk_50',
              'feat_speeddiff_xy', 'feat_speeddiff_xz', 'feat_speeddiff_yz']

    series += [f'position_change_x_{period}' for period in periods]
    series += [f'position_change_y_{period}' for period in periods]
    series += [f'position_change_z_{period}' for period in periods]
    series += [f'position_change_{period}' for period in periods]

    for serie in series:
        for period in periods:
            # Basic aggregations
            rolled_series = df[f'{serie}'].rolling(period)
            if (f'{serie}_{period}_mean' in SELECTED_VARS):
                df[f'{serie}_{period}_mean'] = rolled_series.mean()
            if (f'{serie}_{period}_min' in SELECTED_VARS):
                df[f'{serie}_{period}_min'] = rolled_series.min()
            if (f'{serie}_{period}_max' in SELECTED_VARS):
                df[f'{serie}_{period}_max'] = rolled_series.max()
            if (f'{serie}_{period}_std' in SELECTED_VARS):
                df[f'{serie}_{period}_std'] = rolled_series.std()
            if (f'{serie}_{period}_skew' in SELECTED_VARS):
                df[f'{serie}_{period}_skew'] = rolled_series.skew()
            if (f'{serie}_{period}_kurt' in SELECTED_VARS):
                df[f'{serie}_{period}_kurt'] = rolled_series.kurt()
            if (f'{serie}_{period}_ewm' in SELECTED_VARS):
                df[f'{serie}_{period}_ewm'] = df[f'{serie}'].ewm(min_periods=period, span=period, adjust=False).mean()
            if ((f'{serie}_{period}_trix' in SELECTED_VARS) or
                    (f'{serie}_{period}_t3' in SELECTED_VARS) or
                    (f'{serie}_{period}_dema' in SELECTED_VARS)):
                (df[f'{serie}_{period}_trix'],
                 df[f'{serie}_{period}_t3'],
                 df[f'{serie}_{period}_dema']) = TRIX_T3_DEMA(df[f'{serie}'], period)
            if (f'{serie}_{period}_lin' in SELECTED_VARS):
                df[f'{serie}_{period}_lin'] = ADD_LIN(df[f'{serie}'], period)
            if (f'{serie}_{period}_coef' in SELECTED_VARS):
                df[f'{serie}_{period}_coef'] = ADD_COEF(df[f'{serie}'], period)
            if (f'{serie}_{period}_change' in SELECTED_VARS):
                df[f'{serie}_{period}_change'] = df[f'{serie}'].shift(period) - df[f'{serie}']
            if (f'{serie}_{period}_cmo' in SELECTED_VARS):
                df[f'{serie}_{period}_cmo'] = CMO(df[f'{serie}'], period)
            if (f'{serie}_{period}_rsi' in SELECTED_VARS):
                df[f'{serie}_{period}_rsi'] = RSI(df[f'{serie}'], period)
            if (f'{serie}_{period}_midpoint' in SELECTED_VARS):
                df[f'{serie}_{period}_midpoint'] = MIDPOINT(df[f'{serie}'], period)
            if (f'{serie}_{period}_trima' in SELECTED_VARS):
                df[f'{serie}_{period}_trima'] = TRIMA(df[f'{serie}'], period)
            if (f'{serie}_{period}_closetoclose' in SELECTED_VARS):
                df[f'{serie}_{period}_closetoclose'] = CloseToClose_estimator(df[f'{serie}'] + 1000, period)
            if (f'{serie}_{period}_parkinson' in SELECTED_VARS):
                df[f'{serie}_{period}_parkinson'] = Parkinson_estimator(df[f'{serie}'] + 1000, period)
            if (f'{serie}_{period}_germanklass' in SELECTED_VARS):
                df[f'{serie}_{period}_germanklass'] = GarmanKlass_estimator(df[f'{serie}'] + 1000, period)
            if (f'{serie}_{period}_rogersatchell' in SELECTED_VARS):
                df[f'{serie}_{period}_rogersatchell'] = RogerSatchell_estimator(df[f'{serie}'] + 1000, period)
            if (f'{serie}_{period}_mad' in SELECTED_VARS):
                df[f'{serie}_{period}_mad'] = mad_numpy(df[f'{serie}'].values, period)

    return df


def extract_window_features(df, window_size=26):
    """
    Given a dataframe with 1000 observations extracts features for the 
    last window (i.e. last 26 rows of data)
    """

    features = extract_row_features(df)
    features = extract_historic_features(features)
    features = features.iloc[-window_size:]

    feats_to_drop = [c for c in features.columns if c not in ['session_id', 'timestep', 'region_class'] + SELECTED_VARS]
    features = features.drop(columns=feats_to_drop)

    return features


def extract_all_features(df):
    """
    Function to extract all the features from the raw data iterating
    by session and by window within each session
    """

    # Define parameters of feature extraction
    lookback = 600
    sessions = df['session_id'].unique()
    features = pd.DataFrame()

    # Iterate through every session
    for session in sessions:
        # Filter dataframe of the selected session
        currentdf = df[df['session_id'] == session].reset_index(drop=True)

        # Compute accumulated features from the beginning of the session
        currentdf = extract_position_features(currentdf)

        # Treat each window separately within the filtered session
        current_row = 0
        while current_row < currentdf.shape[0]:
            start = max(current_row - lookback, 0)
            window_size = min(26, currentdf.shape[0] - current_row)
            currentfeatures = extract_window_features(currentdf[start:current_row + window_size], window_size)
            features = pd.concat([features, currentfeatures])
            current_row += window_size

    return features


def train_catboost(X, fold, save_path):
    """
    Trains a catboost model for a dataframe X specifying which rows are used to train with a column
    named 'train' and saves the model object in the specified 'save_path' filename
    
    """

    # Define the target and feature names
    target_name = 'region_class'
    features = list(X.columns)
    features.remove(target_name)
    features.remove('train')
    features.remove('session_id')
    features.remove('timestep')
    
    params = {0: {'learning_rate': 0.008960321310044202, 'max_depth': 6, 'lambda_l2': 5, 'min_data_in_leaf': 3400, 'num_leaves': 250},
              1: {'learning_rate': 0.019751358060036788, 'max_depth': 3, 'lambda_l2': 100, 'min_data_in_leaf': 4900, 'num_leaves': 430},
              2: {'learning_rate': 0.011888935623747047, 'max_depth': 8, 'lambda_l2': 85, 'min_data_in_leaf': 7300, 'num_leaves': 130},
              3: {'learning_rate': 0.018783388135595214, 'max_depth': 6, 'lambda_l2': 80, 'min_data_in_leaf': 3800, 'num_leaves': 370},
              4: {'learning_rate': 0.011186237140744274, 'max_depth': 7, 'lambda_l2': 35, 'min_data_in_leaf': 1700, 'num_leaves': 210}}

    # Define the model
    model = CatBoostClassifier(eta=params[fold]['learning_rate'],
                               n_estimators=10000,
                               task_type='CPU',
                               thread_count=-1,
                               depth=params[fold]['max_depth'],
                               l2_leaf_reg=params[fold]['lambda_l2'],
                               min_data_in_leaf=params[fold]['min_data_in_leaf'],
                               grow_policy='Lossguide',
                               max_leaves=params[fold]['num_leaves'],
                               has_time=True,
                               random_seed=4,
                               loss_function='MultiClass',
                               boosting_type='Plain',
                               class_names=[i for i in range(20)])
    # Fit model
    print('#' * 20 + ' ' * 5 + 'training with ', sum(X['train'] == 1), ' ' * 5 + '#' * 20)
    print('#' * 20 + ' ' * 5 + 'validating with ', sum(X['train'] != 1), ' ' * 5 + '#' * 20)

    model.fit(X[X['train'] == 1][features], 
              X[X['train'] == 1][target_name],
              eval_set=(X[X['train'] != 1][features], 
                        X[X['train'] != 1][target_name]),
              use_best_model=True,
              verbose_eval=100,
              early_stopping_rounds=60)

    # Store the model
    model.save_model(f'{model_filename}_{fold}')

    return model
