import json
import boto3
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder


def load_from_s3(bucket, object_key):
    # Helper function to load objects from S3
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=object_key)
    object_bytes = response['Body'].read()
    return pickle.loads(object_bytes)


def lambda_handler(event, context):
    # Constants
    BUCKET = 'car-price-pred-mlops'

    # Load the model and encoders
    model = load_from_s3(BUCKET, 'finalized_linear_model.pkl')
    scaler = load_from_s3(BUCKET, 'scaler.pkl')
    label_encoder = load_from_s3(BUCKET, 'label_encoder.pkl')
    onehot_encoder = load_from_s3(BUCKET, 'onehot_encoder.pkl')

    # Load incoming JSON data
    car_data = json.loads(event['body'])
    df = pd.DataFrame([car_data])

    # Print initial data frame
    print("Initial DataFrame:", df)

    # Pre-processing
    numerical_columns = ['Kilometeres', 'HorsePower', 'CC', 'Wt', 'Age']
    categorical_columns = ['Fuel_Type', 'Doors', 'Automatic', 'MetallicCol']

    # Apply the scaler
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Apply label encoder
    df['Doors'] = label_encoder.transform(df['Doors'])

    # Apply one-hot encoder
    encoded_features = onehot_encoder.transform(df[['Fuel_Type']])
    encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=onehot_encoder.get_feature_names_out(['Fuel_Type']))
    df = pd.concat([df.drop('Fuel_Type', axis=1), encoded_features_df], axis=1)

    # Ensure 'Automatic' and 'MetallicCol' are treated as categorical
    df['Automatic'] = df['Automatic'].astype('category')
    df['MetallicCol'] = df['MetallicCol'].astype('category')

    # Print processed data frame
    print("Processed DataFrame:", df)

    # Make predictions
    X = df.drop('SellingPrice', axis=1, errors='ignore')  # Ensure 'SellingPrice' is excluded if present
    prediction = model.predict(X)

    # Create JSON response
    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_price': prediction.tolist()})
    }

    return response
