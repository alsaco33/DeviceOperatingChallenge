import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier
import time

SELECTED_VARS = [
    'feat_time',
    'ventile_cat_time',
 'accel_z_10_min',
 'accel_y_5_min',
 'accel_z_10_ewm',
 'accel_y_5_ewm',
 'gyro_z_20_closetoclose',
 'accel_y_10_min',
 'speed_z_100_rsi',
 'accel_y_100_min',
 'gyro_y_100_ewm',
 'accel_y_20_dema',
 'feat_accelcos_yz_5_midpoint',
 'accel_z_5_max',
 'feat_acceldiff_xy_20_dema',
 'gyro_x_100_ewm',
 'gyro_z_100_closetoclose',
 'accel_y_5_midpoint',
 'feat_acceldiff_yz_20_t3',
 'position_z_100_ewm',
 'feat_acceldiff_yz_100_t3',
 'accel_y_20_t3',
 'accel_x_jerk_100_kurt',
 'accel_y_5_mean',
 'accel_z_5_min',
 'gyro_z_100_mean',
 'accel_x_jerk_50_100_ewm',
 'feat_acceldiff_yz_5_min',
 'accel_y_100_rsi',
 'feat_speeddiff_xz_5_trix',
 'accel_y_10_ewm',
 'speed_x_10_closetoclose',
 'gyro_z_100_ewm',
 'speed_z_10_cmo',
 'feat_acceldiff_xy_5_min',
 'sin_position_z_50_min',
 'accel_z_100_max',
 'feat_acceldiff_xz_100_t3',
 'feat_accelangle_xy_20_dema',
 'feat_accelangle_xz_10_trima',
 'sin_position_x_100_rsi',
 'accel_y_5_germanklass',
 'feat_speeddiff_xy_5_cmo',
 'feat_acceldiff_xy_100_rsi',
 'sin_position_z_5_rogersatchell',
 'speed_y_100_coef',
 'sin_position_z_100_rsi',
 'position_x_100_parkinson',
 'accel_y_10_dema',
 'feat_acceldiff_yz_10_min',
 'accel_x_10_max',
 'feat_gyro_normyz_100_trix',
 'accel_z_10_t3',
 'feat_accelsin_xz_20_ewm',
 'feat_acceldiff_yz_20_dema',
 'speed_y_5_cmo',
 'accel_z_5_midpoint',
 'gyro_x_100_mean',
 'feat_acceldiff_xy_20_t3',
 'accel_x_100_t3',
 'accel_y_jerk_20_100_ewm',
 'position_y_50_trix',
 'feat_abs_gyroy_20_ewm',
 'feat_acceldiff_xz_10_max',
 'accel_z_20_min',
 'accel_z_100_t3',
 'feat_speeddiff_xz_100_coef',
 'feat_acceldiff_xy_10_midpoint',
 'feat_accel_normxy_50_ewm',
 'accel_z_50_t3',
 'accel_y_jerk_20_mad',
 'accel_z_10_max',
 'feat_acceldiff_xy_100_trix',
 'feat_speeddiff_yz_100_rsi',
 'feat_acceldiff_yz_10_ewm',
 'accel_x_5_rogersatchell',
 'gyro_y_50_trima',
 'sin_position_x_100_max',
 'accel_z_5_mean',
 'feat_speeddiff_xy_5_trix',
 'feat_acceldiff_xz_5_min',
 'feat_speeddiff_xz_100_rsi',
 'position_change_5_100_min',
 'accel_y_10_max',
 'position_x_100_ewm',
 'accel_x_jerk_50_5_std',
 'feat_accelangle_xz_10_max',
 'cos_position_z_100_rsi',
 'gyro_y_100_mean',
 'gyro_z_100_midpoint',
 'accel_z_20_ewm',
 'speed_y_5_change',
 'accel_y_20_std',
 'speed_z_20_rsi',
 'accel_z_100_ewm',
 'accel_z_20_t3',
 'gyro_x_100_mad',
 'gyro_y_10_closetoclose',
 'feat_speeddiff_xy_10_lin',
 'cos_position_z_100_trix',
 'position_z_100_dema',
 'feat_gyro_normyz_100_std',
 'feat_abs_gyroz_10_midpoint',
 'feat_accel_normyz_100_trix',
 'feat_accel_normxz_20_max',
 'speed_z_20_cmo',
 'sin_position_z_100_midpoint',
 'feat_speeddiff_yz_100_coef',
 'gyro_x_50_trima',
 'sin_position_y_100_max',
 'feat_accel_normxy_50_min',
 'feat_accel_normxz_10_max',
 'feat_acceldiff_yz_100_closetoclose',
 'feat_abs_gyrox_100_mad',
 'gyro_y_20_rsi',
 'feat_accelcos_xy_100_trix',
 'feat_acceldiff_xz_100_min',
 'feat_accelsin_xy_100_rsi',
 'speed_x_5_trix',
 'accel_y_5_max',
 'feat_acceldiff_yz_100_max',
 'gyro_z_100_kurt',
 'sin_position_y_100_coef',
 'sin_position_z_20_rsi',
 'speed_z_5_change',
 'gyro_y_100_kurt',
 'feat_accelangle_xz_100_min',
 'gyro_z_5_rogersatchell',
 'feat_speeddiff_xz_10_cmo',
 'feat_accelangle_xy_20_t3',
 'feat_accelangle_xy_10_min',
 'feat_accel_norm_100_trima',
 'feat_acceldiff_xz_10_min',
 'feat_acceldiff_xy_5_midpoint',
 'sin_position_y_100_rsi',
 'speed_x_100_coef',
 'feat_acceldiff_yz_20_min',
 'feat_accelangle_xz_100_max',
 'sin_position_z_100_max',
 'sin_position_z_100_t3',
 'feat_acceldiff_yz_5_ewm',
 'speed_x_100_trix',
 'feat_accelangle_yz_5_min',
 'feat_gyro_norm_10_dema',
 'feat_accel_norm_100_ewm',
 'position_y_100_parkinson',
 'gyro_y_20_trima',
 'gyro_x_50_rsi',
 'gyro_z_100_t3',
 'accel_x_jerk_100_closetoclose',
 'speed_y_50_lin',
 'gyro_x_100_trima',
 'feat_abs_gyrox_100_trix',
 'gyro_z_50_closetoclose',
 'feat_acceldiff_yz_100_midpoint',
 'feat_acceldiff_yz_100_ewm',
 'gyro_y_100_min',
 'accel_z_5_ewm',
 'sin_position_x_5_rogersatchell',
 'feat_acceldiff_yz_10_dema',
 'gyro_y_20_closetoclose',
 'feat_acceldiff_xy_10_min',
 'feat_accelsin_yz_100_dema',
 'sin_position_y_100_parkinson',
 'feat_accelangle_yz_100_min',
 'cos_position_y_100_trix',
 'feat_accelcos_yz_20_dema',
 'feat_accelangle_xy_10_ewm',
 'feat_accelcos_xz_5_rogersatchell',
 'accel_z_50_ewm',
 'cos_position_y_100_t3',
 'accel_z_50_max',
 'gyro_z_100_dema',
 'speed_y_10_trix',
 'accel_xz_corr_100',
 'feat_accel_normxy_20_min',
 'accel_y_jerk_100_closetoclose',
 'accel_y_100_max',
 'speed_x_100_min',
 'accel_z_100_midpoint',
 'accel_y_50_min',
 'position_change_z_50_100_max',
 'feat_speeddiff_yz_20_rsi',
 'speed_y_100_trix',
 'feat_accel_normyz_20_trix',
 'gyro_z',
 'accel_xz_corr_20',
 'gyro_x_100_max',
 'feat_accelsin_xz_100_dema',
 'feat_acceldiff_xz_10_midpoint',
 'feat_speeddiff_xy_5_parkinson',
 'accel_z_100_rsi',
 'feat_acceldiff_yz_100_min',
 'accel_y_100_midpoint',
 'position_z_100_trix',
 'speed_x_50_trix',
 'feat_speeddiff_xz_10_std',
 'accel_z_50_min',
 'speed_z_100_max',
 'accel_y_50_t3',
 'speed_z_100_min',
 'speed_x_5_rogersatchell',
 'feat_abs_gyrox_100_std',
 'feat_acceldiff_xy_10_max',
 'feat_abs_gyroz_100_trix',
 'accel_yz_corr_10',
 'feat_accel_norm_10_min',
 'feat_abs_accelx_100_trix',
 'feat_acceldiff_yz_20_max',
 'feat_accel_norm_100_mad',
 'position_x',
 'accel_y_jerk_20_100_min',
 'feat_speeddiff_yz_10_rsi',
 'gyro_x_100_min',
 'speed_y_100_max',
 'accel_z_jerk_50_100_ewm',
 'speed_z_100_coef',
 'feat_accelcos_yz_100_ewm',
 'gyro_y_50_mean',
 'gyro_z_50_mean',
 'feat_speeddiff_xy_100_trix',
 'speed_y_50_max',
 'feat_accelsin_xy_20_ewm',
 'feat_accel_norm_100_min',
 'accel_xy_corr_100',
 'feat_speeddiff_xz_100_max',
 'accel_z_100_dema',
 'feat_acceldiff_xy_50_ewm',
 'feat_speeddiff_xz_100_dema',
 'feat_speeddiff_xy_20_closetoclose',
 'cos_position_x_100_rsi',
 'feat_accel_normxz_100_trix',
 'speed_z_5_trix',
 'feat_accel_normxy_100_trix',
 'feat_speeddiff_yz_100_t3',
 'feat_speeddiff_yz_50_dema',
 'speed_x_100_t3',
 'feat_speeddiff_xz_100_min',
 'accel_y_50_midpoint',
 'speed_z_5_cmo',
 'feat_accel_normxy_20_std',
 'position_y_100_trix',
 'speed_y_5_rsi',
 'accel_y_100_dema',
 'feat_acceldiff_xz_20_max',
 'cos_position_y_100_dema',
 'gyro_x_100_t3',
 'feat_acceldiff_yz_50_min',
 'feat_speeddiff_yz_50_rsi',
 'accel_y_jerk_20_100_midpoint',
 'accel_y_50_dema']

SELECTED_VARS += [
 'accel_z_150_min',
 'accel_y_150_min',
 'accel_z_150_ewm',
 'accel_y_150_ewm',
 'gyro_z_150_closetoclose',
 'accel_y_150_min',
 'speed_z_150_rsi',
 'accel_y_150_min',
 'gyro_y_150_ewm',
 'accel_y_150_dema',
 'feat_accelcos_yz_150_midpoint',
 'accel_z_150_max',
 'feat_acceldiff_xy_150_dema',
 'gyro_x_150_ewm',
 'gyro_z_150_closetoclose',
 'accel_y_150_midpoint',
 'feat_acceldiff_yz_150_t3',
 'position_z_150_ewm',
 'feat_acceldiff_yz_150_t3',
 'accel_y_150_t3',
 'accel_x_jerk_150_kurt',
 'accel_y_150_mean',
 'accel_z_150_min',
 'gyro_z_150_mean',
 'accel_x_jerk_50_150_ewm',
 'feat_acceldiff_yz_150_min',
 'accel_y_150_rsi',
 'feat_speeddiff_xz_150_trix',
 'accel_y_150_ewm',
 'speed_x_150_closetoclose',
 'gyro_z_150_ewm',
 'speed_z_10_cmo',
 'feat_acceldiff_xy_150_min',
 'sin_position_z_150_min',
 'accel_z_150_max',
 'feat_acceldiff_xz_150_t3',
 'feat_accelangle_xy_150_dema',
 'feat_accelangle_xz_150_trima',
 'sin_position_x_150_rsi',
 'accel_y_150_germanklass',
 'feat_speeddiff_xy_150_cmo',
 'feat_acceldiff_xy_100_rsi',
 'sin_position_z_150_rogersatchell',
 'speed_y_150_coef',
 'sin_position_z_150_rsi',
 'position_x_150_parkinson',
 'accel_y_150_dema',
 'feat_acceldiff_yz_150_min',
 'accel_x_150_max',
 'feat_gyro_normyz_150_trix',
 'accel_z_150_t3',
 'feat_accelsin_xz_150_ewm',
 'feat_acceldiff_yz_150_dema',
 'speed_y_150_cmo',
 'accel_z_150_midpoint',
 'gyro_x_150_mean',
 'feat_acceldiff_xy_150_t3',
 'accel_x_150_t3',
 'accel_y_jerk_20_150_ewm',
 'position_y_150_trix',
 'feat_abs_gyroy_150_ewm',
 'feat_acceldiff_xz_150_max',
 'accel_z_150_min',
 'accel_z_150_t3',
 'feat_speeddiff_xz_150_coef',
 'feat_acceldiff_xy_150_midpoint',
 'feat_accel_normxy_150_ewm',
 'accel_z_150_t3',
 'accel_y_jerk_150_mad',
 'accel_z_150_max',
 'feat_acceldiff_xy_150_trix',
 'feat_speeddiff_yz_150_rsi',
 'feat_acceldiff_yz_150_ewm',
 'accel_x_150_rogersatchell',
 'gyro_y_150_trima',
 'sin_position_x_150_max',
 'accel_z_150_mean',
 'feat_speeddiff_xy_150_trix',
 'feat_acceldiff_xz_150_min',
 'feat_speeddiff_xz_150_rsi',
 'position_change_5_150_min',
 'accel_y_150_max',
 'position_x_150_ewm',
 'accel_x_jerk_50_150_std',
 'feat_accelangle_xz_150_max',
 'cos_position_z_150_rsi',
 'gyro_y_150_mean',
 'gyro_z_150_midpoint',
 'accel_z_150_ewm',
 'speed_y_150_change',
 'accel_y_150_std',
 'speed_z_150_rsi',
 'accel_z_150_ewm',
 'accel_z_150_t3',
 'gyro_x_150_mad',
 'gyro_y_150_closetoclose',
 'feat_speeddiff_xy_150_lin',
 'cos_position_z_150_trix',
 'position_z_150_dema']


SELECTED_VARS += [
 'accel_z_200_min',
 'accel_y_200_min',
 'accel_z_200_ewm',
 'accel_y_200_ewm',
 'gyro_z_200_closetoclose',
 'accel_y_200_min',
 'speed_z_200_rsi',
 'accel_y_200_min',
 'gyro_y_200_ewm',
 'accel_y_200_dema',
 'feat_accelcos_yz_200_midpoint',
 'accel_z_200_max',
 'feat_acceldiff_xy_200_dema',
 'gyro_x_200_ewm',
 'gyro_z_200_closetoclose',
 'accel_y_200_midpoint',
 'feat_acceldiff_yz_200_t3',
 'position_z_200_ewm',
 'feat_acceldiff_yz_200_t3',
 'accel_y_200_t3',
 'accel_x_jerk_200_kurt',
 'accel_y_200_mean',
 'accel_z_200_min',
 'gyro_z_200_mean',
 'accel_x_jerk_50_200_ewm',
 'feat_acceldiff_yz_200_min',
 'accel_y_200_rsi',
 'feat_speeddiff_xz_200_trix',
 'accel_y_200_ewm',
 'speed_x_200_closetoclose',
 'gyro_z_200_ewm',
 'speed_z_200_cmo',
 'feat_acceldiff_xy_200_min',
 'sin_position_z_200_min',
 'accel_z_200_max',
 'feat_acceldiff_xz_200_t3',
 'feat_accelangle_xy_200_dema',
 'feat_accelangle_xz_200_trima',
 'sin_position_x_200_rsi',
 'accel_y_200_germanklass',
 'feat_speeddiff_xy_200_cmo',
 'feat_acceldiff_xy_200_rsi',
 'sin_position_z_200_rogersatchell',
 'speed_y_200_coef',
 'sin_position_z_200_rsi',
 'position_x_200_parkinson',
 'accel_y_200_dema',
 'feat_acceldiff_yz_200_min',
 'accel_x_200_max',
 'feat_gyro_normyz_200_trix',
 'accel_z_200_t3',
 'feat_accelsin_xz_200_ewm',
 'feat_acceldiff_yz_200_dema',
 'speed_y_200_cmo',
 'accel_z_200_midpoint',
 'gyro_x_200_mean',
 'feat_acceldiff_xy_200_t3',
 'accel_x_200_t3',
 'accel_y_jerk_20_200_ewm',
 'position_y_200_trix',
 'feat_abs_gyroy_200_ewm',
 'feat_acceldiff_xz_200_max',
 'accel_z_200_min',
 'accel_z_200_t3',
 'feat_speeddiff_xz_200_coef',
 'feat_acceldiff_xy_200_midpoint',
 'feat_accel_normxy_200_ewm',
 'accel_z_200_t3',
 'accel_y_jerk_200_mad',
 'accel_z_200_max',
 'feat_acceldiff_xy_200_trix',
 'feat_speeddiff_yz_200_rsi',
 'feat_acceldiff_yz_200_ewm',
 'accel_x_200_rogersatchell',
 'gyro_y_200_trima',
 'sin_position_x_200_max',
 'accel_z_200_mean',
 'feat_speeddiff_xy_200_trix',
 'feat_acceldiff_xz_200_min',
 'feat_speeddiff_xz_200_rsi',
 'position_change_5_200_min',
 'accel_y_200_max',
 'position_x_200_ewm',
 'accel_x_jerk_50_200_std',
 'feat_accelangle_xz_200_max',
 'cos_position_z_200_rsi',
 'gyro_y_200_mean',
 'gyro_z_200_midpoint',
 'accel_z_200_ewm',
 'speed_y_200_change',
 'accel_y_200_std',
 'speed_z_200_rsi',
 'accel_z_200_ewm',
 'accel_z_200_t3',
 'gyro_x_200_mad',
 'gyro_y_200_closetoclose',
 'feat_speeddiff_xy_200_lin',
 'cos_position_z_200_trix',
 'position_z_200_dema']



SELECTED_VARS = list(set(SELECTED_VARS))

def ventile_time(x):
    cuts = [178, 356, 535, 713, 891, 1070, 1249, 1428, 1607, 1787, 
            1968, 2150, 2335, 2524, 2718, 2920, 3130, 3363, 4114]
    for i in range(len(cuts)):
        if x < cuts[i]:
            return i
    return len(cuts)

def TRIX_T3_DEMA(close, period):
    # Triangular and triple 
    ema1 = close.ewm(min_periods = period, span = period, adjust = False).mean()
    ema2 = ema1.ewm(min_periods = period, span = period, adjust = False).mean()
    ema3 = ema2.ewm(min_periods = period, span = period, adjust = False).mean()
    t3 = 3*ema1 - 3*ema2 + ema3
    trix = 100*(ema3-ema3.shift()) / (ema3+ 0.0001)
    dema = 2*ema1 - ema2
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
    cmo = 100 * (ups-downs)/(ups+downs+0.0001)
    return cmo

def RSI(close, period):
    # Relative strength index
    close_delta = close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com = period - 1, adjust=True, min_periods = period).mean()
    ma_down = down.ewm(com = period - 1, adjust=True, min_periods = period).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))    
    return rsi

def MIDPOINT(close, period):
    return (close.rolling(period).max() + close.rolling(period).min())/2

def TRIMA(close, period):
    # Triangular moving average
    nm = round((period+1)/2)
    SMAm = close.rolling(nm).mean()
    trima = SMAm.rolling(nm).mean()
    return trima

def CloseToClose_estimator(close, period): 
    log_return = (close / close.shift(1)).apply(np.log)
    result = log_return.rolling(window=period,center=False).std()
    return result

def Parkinson_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    rs = (1.0 / (4.0 * np.log(2.0))) * ((High / Low).apply(np.log))**2.0
    result = rs.rolling(window=10).mean()
    return result
    
def GarmanKlass_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    Open = close.shift(period)
    
    log_hl = (High / Low).apply(np.log)
    log_co = (close / Open).apply(np.log)
    rs = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
    result = rs.rolling(window=10).mean()
    return result
    
def RogerSatchell_estimator(close, period):
    High = close.rolling(period).max()
    Low = close.rolling(period).min()
    Open = close.shift(period)
    log_ho = (High/ Open).apply(np.log)
    log_lo = (Low / Open).apply(np.log)
    log_co = (close /Open).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    result = rs.rolling(window=10).mean()
    return result

def strided_app(a, L, S ):  
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def mad_numpy(a, W):
    if W>len(a):
        return [np.nan for i in range(len(a))]
    a2D = strided_app(a,W,1)
    mad = np.absolute(a2D - moving_average(a,W)[:,None]).mean(1)
    mad = np.concatenate([[np.nan for i in range(W-1)], mad])
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
    df['sin_position_x'] = np.sin(df['position_x']/360*2*np.pi)
    df['sin_position_y'] = np.sin(df['position_y']/360*2*np.pi)
    df['sin_position_z'] = np.sin(df['position_z']/360*2*np.pi)
    
    # Extract cos of the position coordinates
    df['cos_position_x'] = np.cos(df['position_x']/360*2*np.pi)
    df['cos_position_y'] = np.cos(df['position_y']/360*2*np.pi)
    df['cos_position_z'] = np.cos(df['position_z']/360*2*np.pi)
    
    # Extract speed features
    df['speed_x'] = (df['accel_x']/0.0312319993972778 -15).cumsum() / units_per_sec
    df['speed_y'] = (df['accel_y']/0.0312319993972778 +4).cumsum()  / units_per_sec
    df['speed_z'] = (df['accel_z']/0.0312319993972778 +3).cumsum()  / units_per_sec
    
    return df

def extract_row_features(df):
    """
    
    """
    
    # Extract time from timestep
    df['feat_time'] = df['timestep'].apply(lambda x: str(x).replace('timestep_', '')).astype(int)
    df['ventile_cat_time'] = df['feat_time'].apply(lambda x: ventile_time(x)).astype(int)
    
    # Extract norms of the acceleration
    df[f'feat_accel_norm'] = np.sqrt(df[f'accel_x']**2 + df[f'accel_y']**2 + df[f'accel_z']**2)
    df[f'feat_accel_normxy'] = np.sqrt(df[f'accel_x']**2 + df[f'accel_y']**2)
    df[f'feat_accel_normxz'] = np.sqrt(df[f'accel_x']**2 + df[f'accel_z']**2)
    df[f'feat_accel_normyz'] = np.sqrt(df[f'accel_y']**2 + df[f'accel_z']**2)
    
    # Extract norms of the gyro
    df[f'feat_gyro_norm'] = np.sqrt(df[f'gyro_x']**2 + df[f'gyro_y']**2 + df[f'gyro_z']**2)
    df[f'feat_gyro_normxy'] = np.sqrt(df[f'gyro_x']**2 + df[f'gyro_y']**2)
    df[f'feat_gyro_normxz'] = np.sqrt(df[f'gyro_x']**2 + df[f'gyro_z']**2)
    df[f'feat_gyro_normyz'] = np.sqrt(df[f'gyro_y']**2 + df[f'gyro_z']**2)
    
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
    """
    
    """
    
    df['accel_x_jerk'] = df['accel_x'].diff().fillna(0)
    df['accel_y_jerk'] = df['accel_y'].diff().fillna(0)
    df['accel_z_jerk'] = df['accel_z'].diff().fillna(0)
    
    df['accel_x_jerk_20'] = (df['accel_x']-df['accel_x'].rolling(20).mean()).fillna(0)
    df['accel_y_jerk_20'] = (df['accel_y']-df['accel_y'].rolling(20).mean()).fillna(0)
    df['accel_z_jerk_20'] = (df['accel_z']-df['accel_z'].rolling(20).mean()).fillna(0)
    
    df['accel_x_jerk_50'] = (df['accel_x']-df['accel_x'].rolling(50).mean()).fillna(0)
    df['accel_y_jerk_50'] = (df['accel_y']-df['accel_y'].rolling(50).mean()).fillna(0)
    df['accel_z_jerk_50'] = (df['accel_z']-df['accel_z'].rolling(50).mean()).fillna(0)
    
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
        df[f'position_change_x_{period}'] = ((df['sin_position_x'] - df['sin_position_x'].shift(period))**2).fillna(0)
        df[f'position_change_y_{period}'] = ((df['sin_position_y'] - df['sin_position_y'].shift(period))**2).fillna(0)
        df[f'position_change_z_{period}'] = ((df['sin_position_z'] - df['sin_position_z'].shift(period))**2).fillna(0)
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
              'sin_position_x',  'sin_position_y', 'sin_position_z',
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
    features = features.drop(columns = feats_to_drop)
    
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
        currentdf = df[df['session_id']==session].reset_index(drop=True)
        
        # Compute accumulated features from the beginning of the session
        currentdf = extract_position_features(currentdf)
        
        # Treat each window separately within the filtered session
        current_row = 0
        while current_row < currentdf.shape[0]:
            start = max(current_row - lookback, 0)
            window_size = min(26, currentdf.shape[0] - current_row)
            currentfeatures = extract_window_features(currentdf[start:current_row+window_size], window_size)
            features = pd.concat([features, currentfeatures])
            current_row += window_size
            
    return features

def train_catboost(X, save_path):
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
    
    # Define the model
    model = CatBoostClassifier(eta=0.01,
                               n_estimators=10,
                               task_type='CPU',
                               thread_count=-1,
                               depth=4,
                               l2_leaf_reg=20,
                               min_data_in_leaf=1000,
                               grow_policy='Lossguide',
                               max_leaves=20,
                               has_time=True,
                               random_seed=4,
                               loss_function='MultiClass',
                               boosting_type='Plain',
                               class_names=[i for i in range(20)])
    # Fit model
    print('#'*20 + ' '*5 + 'training with ', sum(X['train'] == 1), ' '*5 + '#'*20)
    print('#'*20 + ' '*5 + 'validating with ', sum(X['train'] != 1), ' '*5 + '#'*20)
    
    model.fit(X[X['train'] == 1][features], X[X['train'] == 1][target_name],
              eval_set=(X[X['train'] != 1][features], X[X['train'] != 1][target_name]),
              use_best_model=True,
              verbose_eval=5,
              early_stopping_rounds=20)

    # Store the model
    model.save_model(save_path)
            
    return model