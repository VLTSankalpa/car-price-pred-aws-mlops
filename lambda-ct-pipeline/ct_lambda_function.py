import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from io import BytesIO

# Initialize clients
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # Get bucket and file details from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Get the file from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(response['Body'])

    # Data Preprocessing
    df = df.drop('Unnamed: 0', axis=1)
    numerical_columns = ['Kilometeres', 'HorsePower', 'CC', 'Wt', 'Age']
    categorical_columns = ['Fuel_Type', 'Doors', 'Automatic', 'MetallicCol']
    df[numerical_columns] = MinMaxScaler().fit_transform(df[numerical_columns])
    df['Doors'] = LabelEncoder().fit_transform(df['Doors'])
    df = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=False, dtype=int)
    df['Automatic'] = df['Automatic'].astype('category')
    df['MetallicCol'] = df['MetallicCol'].astype('category')

    # Feature and label separation
    X = df.drop(['SellingPrice'], axis=1)
    y = df['SellingPrice']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model Evaluation
    y_val_pred = model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    mae_val = mean_squared_error(y_val, y_val_pred, squared=False)
    mape_val = np.mean(np.abs((y_val - y_val_pred) / y_val).replace(np.inf, np.nan)) * 100
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=True)

    # Save the model to S3
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    s3_client.put_object(Bucket='your-model-bucket', Key='finalized_linear_model.pkl', Body=buffer)

    return {
        'statusCode': 200,
        'body': {
            'mse': mse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'rmse': rmse_val,
            'message': 'Model trained and saved successfully'
        }
    }

