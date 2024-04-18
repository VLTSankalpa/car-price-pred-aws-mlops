# **Car Price Prediction - MLOps on AWS**

This repository contains the resources and source code for a machine learning assignment aimed at predicting car prices. The project leverages AWS services, including Lambda and API Gateway, and follows MLOps practices to automate and monitor all steps of machine learning system construction.

# **Repository Structure**

```python
pythonCopy code
.
├── README.md
├── data
│   └── dataset.npz              # Raw and preprocessed data
├── images
│   └── kde.png                  # Images used in README documentation
├── lambda-ct-pipeline
│   └── ct_lambda_function.py    # Lambda function for continuous training pipeline
├── lambda-model-endpoint
│   ├── Dockerfile               # Dockerfile for building Lambda deployment image
│   ├── main.py                  # Main script for Lambda function initialization
│   └── model_endpoint_lambda_function.py  # Lambda function for model predictions
├── model
│   ├── finalized_linear_model.pkl  # Saved final linear model
│   ├── label_encoder.pkl           # Label encoder for categorical data preprocessing
│   ├── model.py                    # Script for model training and evaluation
│   ├── onehot_encoder.pkl          # One-hot encoder for categorical data preprocessing
│   ├── scaler.pkl                  # Scaler object for numerical data normalization
│   └── train.csv                   # Training dataset
└── notebooks
    └── development-notebook.ipynb  # Jupyter notebook containing EDA, data visualization, data preprocessing, and model development

```

### **Directories and Files**

- **`/data`**: Contains raw and preprocessed datasets used in model training.
- **`/images`**: Includes images used within the README documentation to explain concepts or results.
- **`/lambda-ct-pipeline`**: Holds the AWS Lambda function for continuous training of the machine learning model.
- **`/lambda-model-endpoint`**:
   - **`Dockerfile`**: Defines the Docker container used to deploy the Lambda function.
   - **`test.py`**: Unit test for the model endpoint Lambda function.
   - **`model_endpoint_lambda_function.py`**: Implements the Lambda function to serve the model predictions.
- **`/model`**: Contains all machine learning models and their corresponding encoders, along with the training dataset.
   - **`finalized_linear_model.pkl`**: The serialized final linear regression model ready for predictions.
   - **`label_encoder.pkl`**, **`onehot_encoder.pkl`**, **`scaler.pkl`**: Serialization of preprocessing encoders.
   - **`model.py`**: Initial model training python code provided by the instructor.
- **`/notebooks`**: Jupyter notebook that documents the exploratory data analysis (EDA), data visualization, data preprocessing, model development with refinements.

# Local Development Setup

For this assignment, I set up the local development environment on a Mac M1 Pro using Miniconda. I followed these specific steps to ensure that all necessary Python packages and Jupyter functionalities were available:

1. **Create a Conda Environment**:

    ```bash
    conda create --name car-price-pred-mlops python
    ```

   This command creates a new Conda environment named **`car-price-pred-mlops`**. It's isolated from other environments, ensuring that package dependencies do not interfere with those in different projects.

2. **Activate the Environment**:

    ```bash
    conda activate car-price-pred-mlops
    ```

   Activating the environment makes it the current working environment, which means all Python and command-line operations take place within this environment.

3. **Install Jupyter Notebook**:

    ```bash
    conda install jupyter
    ```

   Jupyter Notebook is installed in the environment, which is a powerful tool for interactive coding and visualization, often used in data science and machine learning projects.

4. **Install IPython Kernel**:

    ```bash
    conda install ipykernel
    ```

   This command installs the IPython kernel, which allows Jupyter to run Python code. The kernel acts as the backend that processes the code written in the notebook.

5. **Set up IPython Kernel for the Environment**:

    ```bash
    python -m ipykernel install --user --name car-price-pred-mlops --display-name "Car Price Prediction MLOps"
    ```

   This step registers the newly created Conda environment in Jupyter under the name "Car Price Prediction MLOps". It allows the environment to be selected as the kernel in Jupyter notebooks, ensuring that notebooks use the specific environment's settings and installed packages.

6. **Launch Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

   This command starts the Jupyter Notebook server locally in web browser, from where we can create and manage your notebooks.

## **Additional Requirements**

1. **Docker Installation**:
    - Docker is installed to handle containerization, necessary for deploying functions and services.
    - Visit Docker's official site for installation instructions.
2. **AWS CLI Installation**:
    - The AWS Command Line Interface (CLI) is installed to interact with AWS services directly from the terminal.
    - Detailed installation guides are available on the [AWS documentation page](https://aws.amazon.com/cli/).
3. **AWS CLI Configuration**:
    - AWS CLI is configured with user credentials to authenticate and interact with AWS resources.
    - Run **`aws configure`** to set up AWS access key ID, secret access key, region, and output format.
4. **Python Libraries Installation**:
    - Essential Python libraries are required for data handling, statistical analysis, and machine learning operations.
    - To install these libraries, use the **`requirements.txt`** file provided in the repository.

        ```bash
        pip install -r requirements.txt
        ```

# **Step 1: Model Preparation for Deployment**

## **Exploratory Data Analysis (EDA)**

In this phase, I performed Exploratory Data Analysis (EDA) on the Car Price Prediction dataset using a suite of standard templates that I have developed. 
This practice is standard whenever I approach a data science problem. 
The initial investigation and cleaning process are crucial for refining the provided Python script. 
This involves adding the necessary preprocessing steps required for creating an accurate predictive model. 
The goal of this phase is to prepare a clean and well-understood dataset, ensuring that it is ready for subsequent feature engineering and model development steps.

### EDA Objectives
The following key tasks are executed to achieve a comprehensive understanding and preparation of the dataset:

- **List of Columns**: Identify all columns in the dataset to understand the features available.
- **Dataset Shape**: Determine the size of the dataset to understand the scope of data.
- **Data Types**: Ascertain the data types of each column to identify any necessary conversions.
- **Unique Values**: List all unique values in each column to detect any anomalies or irregularities.
- **Convert Data Types**: Adjust the data types of specific columns as necessary for proper analysis.
- **Handling Missing Values**: Identify and address any missing data in the dataset.
- **Summary Statistics**: Generate summary statistics of numeric columns to gain insights into the distribution and central tendencies of the data.

### Insights for EDA

**Categorical Variables**

In the given dataset, the following variables are considered categorical as these variables are used to group data into categories where each category is distinct and has no inherent numerical relationship with the others.

- **Fuel_Type**: Includes categories like 'Diesel', 'Petrol', and 'CNG'. Used as categorical due to distinct fuel types influencing car performance and pricing.
- **Doors**: Represents the number of doors (2, 3, 4, 5) and is treated as categorical to differentiate car body styles.
- **Automatic**: Binary variable (0 for manual, 1 for automatic), distinguishing between transmission types.
- **MetallicCol**: Binary variable (0 for non-metallic, 1 for metallic), indicating the presence of a metallic paint finish.


**Numerical Variables:**

Below variables can be identified as numerical variable as they provide quantitative measures that are essential for calculations and model estimations in machine learning.

- **Kilometers**: A continuous variable showing the car's mileage, which directly influences car value and usage characteristics.
- **HorsePower**: Measures the engine power in horsepower, a continuous quantity impacting car performance.
- **CC**: Engine capacity in cubic centimeters, treated as numerical to quantify engine size.
- **Wt** (Weight): The weight of the car in kilograms, a continuous numerical measure relevant to vehicle dynamics and efficiency.
- **SellingPrice**: Often the target variable, representing the price at which the car is sold, treated as a continuous numerical variable.
- **Age**: Represents the age of the car in years. Although it could be ordinal, it is treated as numerical due to its direct quantitative impact on the car's value and condition.


## Data Visualization

Data visualization is a crucial phase in data analysis for identifying patterns, relationships, and gathering insights that may not be apparent from raw data alone. Effective visualizations can illuminate trends and provide a clearer understanding of the data set's dynamics, particularly in relation to car price. To comprehensively explore the data, various visualization techniques are employed, each serving distinct purposes in the analysis:

- **Kernel Density Estimate (KDE) Plots**: Useful for understanding the distribution of numerical data.
- **Q-Q Plots**: Help assess if a dataset is distributed a certain way, typically gaussian.
- **Histograms**: Ideal for visualizing the distribution of data and observing the shape.
- **Boxplots**: Provide a graphical representation of the numerical data through their quartiles and are especially useful for detecting outliers.
- **Scatter Plots**: Highlight correlations or dependencies between two variables.
- **Heatmaps**: Useful for visualizing the correlation matrix of variables.
- **Count Plots**: Excellent for visualizing categorical data distributions.


### Plotting Detailed Visualizations
To further analyze and visualize the data, specific Python code leveraging `matplotlib` and `seaborn` libraries is employed. The code snippets enhance the aesthetics and provide detailed insight for each numerical column in the dataset.

### Insights Derived from KDE Plots

![KDE Plots](images/kde.png)
Initial observations from Kernel Density Estimate (KDE) plots include:
1. **Kilometers**: The KDE plot for Kilometres shows a right-skewed distribution, indicating that most of the cars have lower mileage, with few cars having high mileage.
2. **HorsePower**: The HorsePower distribution is multi-modal with specific peaks, suggesting clusters around common horsepower ratings for cars.
3. **CC (Engine Size)**: The distribution shows multiple peaks, which may indicate common engine sizes. It is right-skewed, suggesting smaller engine sizes are more prevalent.
4. **WT (Weight)**: The weight distribution appears normally distributed with a single peak, indicating most cars have a weight around this central value.
5. **SellingPrice**: The SellingPrice is right-skewed, suggesting that most cars are clustered around a lower price range with fewer high-priced cars.
6. **Age**: The car age distribution is also right-skewed, indicating that there are more newer cars and fewer older cars.

## Data Preprocessing

### Normalization
To handle potential scale discrepancies among these numerical features, we use the MinMaxScaler from scikit-learn. This scaler transforms each feature to a range between 0 and 1, maintaining the distribution but aligning the scales. This is crucial as it prevents attributes with larger ranges from dominating those with smaller ranges, which is important for many machine learning algorithms.

### **Label Encoding**

For the **`Doors`** feature, which is ordinal, we apply label encoding. This approach converts the categorical labels into a single integer column, preserving the order, which is appropriate for ordinal data.

### **One-Hot Encoding**

The **`Fuel_Type`** feature is treated with one-hot encoding, which is essential for nominal categorical data. This method transforms each categorical value into a new binary column, ensuring that the model interprets these attributes correctly without any implicit ordering.

### **Feature Transformation**

After encoding, we handle the transformation from sparse to dense formats. Many machine learning algorithms require a dense matrix format, so we convert the sparse matrix obtained from one-hot encoding into a dense format. This is performed using the **`.toarray()`** method, which is necessary to integrate these features into the main DataFrame seamlessly.

### **Integration with Original DataFrame**

The newly created dense matrix columns are named according to the unique values in **`Fuel_Type`** and then concatenated back to the original DataFrame. Columns derived from **`Fuel_Type`** are added, and the original **`Fuel_Type`** column is dropped to avoid redundancy.

### **Final Adjustments**

For binary categorical features like **`Automatic`** and **`MetallicCol`**, which are already in a binary format, we explicitly cast them to a 'category' type to ensure consistency in data types across the DataFrame. This step is important for some types of statistical analysis and modeling in Python.

### **Training Data Preparing**

This code performs the following operations:

- [ ] Splits the data into feature (**`X`**) and label (**`y`**) arrays.
- [ ] Uses **`train_test_split`** twice to create a train set (60% of the data), a validation set (20%), and a test set (20%).
- [ ] Saves the training, validation, and test sets to an **`.npz`** file, which can then be loaded for training.
### **Model Training**

Improvement have done to the provided linear regression model is codes for training and evaluating the model. The model is trained on the training set and evaluated on the validation set.

### **Model Evaluation**

To evaluate the performance of the trained model, the following metrics are calculated using the validation set:

- **Mean Squared Error (MSE)**: Represents the average of the squares of the errors—i.e., the average squared difference between the estimated values and the actual value.
- **R-Squared (R²)**: Provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Mean Absolute Percentage Error (MAPE)**: Measures the accuracy as a percentage, and is commonly used to forecast error in predictive modeling.
- **Root Mean Squared Error (RMSE)**: The square root of the mean of the squared errors; RMSE is a good measure of how accurately the model predicts the response.





# Monitoring and Observability

To effectively monitor and observe the AWS Lambda function's performance and behavior, following steps of integrating it with AWS CloudWatch for metrics, logs, and alerts is crucial. This setup provides visibility into the function's operation, helps identify performance bottlenecks, and alerts to potential issues.

### **1. Enable CloudWatch Logs for Lambda Function**

AWS Lambda automatically monitors functions, reporting metrics through Amazon CloudWatch. We just have to ensure **logging is enabled in the Lambda function’s IAM role**. This role needs permission to write logs to CloudWatch. The necessary policy (**`AWSLambdaBasicExecutionRole`**) includes permissions for logs creation.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2f7830c3-173b-41cd-aa45-25fca7558acb/1c04c73e-74cf-4599-a6a6-36369b7187f1/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2f7830c3-173b-41cd-aa45-25fca7558acb/d8b5a216-2123-4208-b5f7-fae373031882/Untitled.png)

- The **`print`** statements in Lambda python function will direct these logs to CloudWatch under the **`/aws/lambda/model-endpoint-v2`** log group.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2f7830c3-173b-41cd-aa45-25fca7558acb/f849a7ac-b3e5-4407-a756-70ab86d7a5db/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2f7830c3-173b-41cd-aa45-25fca7558acb/42a948c8-b161-4173-8209-96065ee0e0ba/Untitled.png)

### **2. Monitor Execution Time and Invocation Frequency**

- **CloudWatch Metrics**: AWS Lambda automatically sends these metrics to CloudWatch:
    - **`Duration`**: Measures the elapsed runtime of your Lambda function in milliseconds.
    - **`Invocations`**: Counts each time a function is invoked in response to an event or invocation API call.

These metrics are found in the CloudWatch console under the **Metrics** section.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2f7830c3-173b-41cd-aa45-25fca7558acb/4dc54515-1733-417c-a018-0b7e48e68e21/Untitled.png)

### **3. Monitor Model Inference Errors**

- **Custom Metrics**: If your model throws specific errors (e.g., inference errors), you might want to log these explicitly and create custom CloudWatch metrics using these logs.
- **Implement Error Handling in Lambda Code**:

    ```python
    import logging
    import boto3
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    cloudwatch = boto3.client('cloudwatch')
    
    def lambda_handler(event, context):
        try:
            # Your model inference code
        except Exception as e:
            logger.error("Model inference failed: %s", str(e))
            cloudwatch.put_metric_data(
                MetricData=[
                    {
                        'MetricName': 'ModelInferenceErrors',
                        'Dimensions': [
                            {'Name': 'FunctionName', 'Value': context.function_name}
                        ],
                        'Unit': 'Count',
                        'Value': 1
                    },
                ],
                Namespace='MyApp/Lambda'
            )
            raise
    ```

### **4. Set Up CloudWatch Alerts**

- **Create CloudWatch Alarms**: Use these to get notified about issues like high latency or increasing error rates.
    - Go to the CloudWatch console → Alarms → Create alarm.
    - Select the metric (e.g., **`Duration`**, **`Errors`**), specify the threshold (e.g., Duration > 3000 ms), and set the period over which this is measured.
    - Configure actions to notify you via SNS (Simple Notification Service) when the alarm state is triggered.