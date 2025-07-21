if [ $# -ne 10 ]; then
    echo "Invalid number of parameters, you need to provide role name, region name, bucket name, prefix, express region name and express bucket name, custom endpoint for s3 standard, auth profile arn and buckets names for testing auth profile"
    echo "Usage: $0 S3RoleName us-west-2 s3torchconnector-test-bucket-name prefix-name/ us-east-1 s3torchconnectorclient-express-bucket-name https://s3.amazon.com arn:aws:iam::XXXXXXXXXXX:role/RoleName profile-test-bucket-name profile-test-express-bucket-name "
    exit 1
fi

ROLE_NAME=$1
REGION_NAME=$2
BUCKET_NAME=$3
PREFIX=$4
EXPRESS_REGION_NAME=$5
EXPRESS_BUCKET_NAME=$6
S3_CUSTOM_ENDPOINT_URL=$7
PROFILE_IAM_ROLE=$8
S3_PROFILE_BUCKET=$9
S3_EXPRESS_PROFILE_BUCKET=${10}

FILE_NAME="tmp_cred.json"
# Set metadata token TTL to 6 hours
TOKEN=`curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
# Retrieve temporary credentials and save to file
curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/${ROLE_NAME} >> ${FILE_NAME}
# Expose temporary credentials to use from cibuildwheel container
export AWS_ACCESS_KEY_ID=`cat ${FILE_NAME} | jq -r '.AccessKeyId'`
export AWS_SECRET_ACCESS_KEY=`cat ${FILE_NAME} | jq -r '.SecretAccessKey'`
export AWS_SESSION_TOKEN=`cat ${FILE_NAME} | jq -r '.Token'`
rm ${FILE_NAME}

# Expose settings for integration tests to use from cibuildwheel container
export S3_REGION=${REGION_NAME}
export S3_BUCKET=${BUCKET_NAME}
export S3_PREFIX=${PREFIX}
export S3_EXPRESS_REGION=${EXPRESS_REGION_NAME}
export S3_EXPRESS_BUCKET=${EXPRESS_BUCKET_NAME}
export S3_CUSTOM_ENDPOINT_URL=${S3_CUSTOM_ENDPOINT_URL}
export PROFILE_IAM_ROLE=${PROFILE_IAM_ROLE}
export S3_PROFILE_BUCKET=${S3_PROFILE_BUCKET}
export S3_EXPRESS_PROFILE_BUCKET=${S3_EXPRESS_PROFILE_BUCKET}

cibuildwheel --output-dir wheelhouse --platform linux s3torchconnectorclient
