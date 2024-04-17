import json
import sys
import os
sys.path.append(os.path.abspath('/Users/tharindu/git/car-price-pred-mlops/lambda-model-endpoint'))
from model_endpoint_lambda_function import lambda_handler  # Adjust import according to your actual module name


def main():
    # Sample test data event
    event = {
        'body': json.dumps({
            'Kilometeres': 45000,
            'Doors': 2,
            'Automatic': 0,
            'HorsePower': 110,
            'MetallicCol': 1,
            'CC': 1500,
            'Wt': 950,
            'Age': 2,
            'Fuel_Type': 'Diesel'
        })
    }

    # Invoke the lambda handler
    response = lambda_handler(event)
    print("Response:", response)


if __name__ == "__main__":
    main()

