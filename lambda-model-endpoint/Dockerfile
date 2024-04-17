# Use the AWS provided base image for Python 3.8
FROM public.ecr.aws/lambda/python:3.8

# Copy function code and any additional files
COPY . ${LAMBDA_TASK_ROOT}

# Install OS packages if necessary
RUN yum install -y gcc-c++

# Install Python dependencies
RUN pip install --no-cache-dir boto3 pandas scikit-learn

# Set the CMD to your handler (this could be the file name and the function handler)
CMD ["model_endpoint_lambda_function.lambda_handler"]
