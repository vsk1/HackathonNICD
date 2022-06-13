import pandas as pd 
import numpy as np 
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


df = pd.read_csv("D:/Hackathon2022/Decarbonising_Supply_Chain/HackathonNICD/Data/1. Mandata Vehicle Summary.csv")
df = df.dropna()
df = df.reset_index(drop = True)

df_annual = pd.read_csv("D:/Hackathon2022/Decarbonising_Supply_Chain/HackathonNICD/Data/1a. Mandata Vehicle Annual Summary.csv")

def duration_minutes(df): 
    df= df.str.split(":").apply(lambda x: int(x[0]) * 60 + int(x[1]))
    
    return df

def date_time_extract(df, col):
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day
    df[col + "_weekday"] = df[col].dt.weekday
    df[col + '_hour'] = df[col].dt.hour


def preprocessing(df, df_annual):
    col_min = ["DRIVING_TIME", "IDLE_TIME", "STANDING_TIME", "INACTIVE_TIME", "UTILISATION"]
    df[col_min] = df[col_min].apply(duration_minutes)
    df["mileage"] = df["KM"] / df["LITRES"]
    df.loc[~np.isfinite(df['mileage']), 'mileage'] = np.nan
    df[df.isnull().any(axis=1)]
    df["mileage"].fillna(df["mileage"].mean(), inplace=True)
    df_annual[["UTILISATION_TIME"]] = df_annual[["UTILISATION_TIME"]].apply(duration_minutes)
    df_annual["util_min"] = df_annual["CO2_KG"] / df_annual["UTILISATION_TIME"] 
    df.drop(["ODOMETER_START", "DATE", "ODOMETER_END", "JOURNEYS", "GALLONS", "MPG", "MILES", "L_100KM", "STANDING_PERC", "INACTIVE_PERC", 
    "UTILISATION_PERC", "IDLE_PERC", "DRIVE_PERC"], axis = 1, inplace = True, errors="ignore")

    df["START"] = pd.to_datetime(df["START"])
    df["END"] = pd.to_datetime(df["END"])
    dfannual_clip = df_annual[["REG","util_min"]]
    df_final = pd.merge(df, dfannual_clip, on="REG", how="outer")
    df_final["CO2"] = df_final["util_min"] * df_final["UTILISATION"]
    df_final.drop(["INACTIVE_TIME", "util_min"], axis = 1, inplace=True, errors="ignore")
    date_time_extract(df_final, "START")
    date_time_extract(df_final, "END")
    df_final.drop(["START_year", "END_year", "START", "END", "DATE"], axis = 1, inplace = True, errors = "ignore")

    return df_final

df_final = preprocessing(df, df_annual)

x = df_final.drop('CO2', axis = 1)
y = df_final['CO2']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=42)

col_to_scale = ["DRIVING_TIME", "IDLE_TIME", "STANDING_TIME", "KM", "LITRES", "mileage"]
cat_col = ["REG"]

numeric_transformer = Pipeline(
    steps=[("scaler", MinMaxScaler())])

categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "most_frequent")), 
("OneHotEncoder", OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(
    transformers = [
                    ("num", numeric_transformer, col_to_scale),
                    ("cat", categorical_transformer, cat_col),
                    ]
)


reg = Pipeline(
    steps = [("preprocessor", preprocessor), 
    ("classifier", XGBRegressor(objective ='reg:squarederror', colsample_bytree=0.9, learning_rate=0.5))]
)

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

filename = 'finalized_model.sav'
pickle.dump(reg, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

print(f"The R2 score is: {r2_score(y_test, y_pred)}")
print(f"The Mean Average Error is: {mean_absolute_error(y_test, y_pred)}")
print(f"The Mean Squared Error is: {mean_squared_error(y_test, y_pred)}")
print(f"The Root Mean Squared Error is: {np.sqrt(mean_squared_error(y_test, y_pred))}")











