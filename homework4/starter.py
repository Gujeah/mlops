

import pickle
import pandas as pd
import os
import argparse

#Parametrice the script via command line interface (variables that are determined by user input at runtime)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



categorical = ['PULocationID', 'DOLocationID']
def wrangle(year, month):
    url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df=pd.read_parquet(url)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df
    

year=2023
month=3
df=wrangle(year, month)
df.head(3)




dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


predictions=y_pred

y_pred.std()


df_result=pd.DataFrame( {
    'ride_id': df['ride_id'],
    'prediction': predictions  
}
)
df_result.head()


output_file = f'yellow_tripdata_{year}-{month:02d}_predictions.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',   
    compression=None,  
    index=False          
)


print(f"\nPredictions saved to {output_file}")




file_size_bytes = os.path.getsize(output_file)
file_size_mb = file_size_bytes / (1024 * 1024) 

print(f"Size of the output file ({output_file}): {file_size_mb:.2f} MB")


##writing the main script 
if __name__=="__main__":
   parser=argparse.ArgumentParser( 
      description="Running New york taxi prediction duration"
   )
   parser.add_argument( 
      '--year',
      type=int,
      required=True,
      help='year of taxi trip data e.g, 2023, 2022, 2035'

   )
   parser.add_argument( 
      '--month',
      type=int,
      required=True,
      help="month of the year from 1-12"
   )
   args=parser.parse_args()
   year=args.year
   month=args.month

   print(f"Starting prediction process for Year: {year}, Month: {month:02d}")
   df = wrangle(year, month)
   print(f"Data for {year}-{month:02d} loaded and processed. Rows: {len(df)}")

    # Preparing features for prediction
   dicts = df[categorical].to_dict(orient='records')
   X_val = dv.transform(dicts)
   print("Features transformed for prediction.")
    
   y_pred = model.predict(X_val)
   print("Predictions generated.")
   mean_predicted_duration = y_pred.mean()
   print(f"Mean predicted duration: {mean_predicted_duration:.2f} minutes")
    

  




