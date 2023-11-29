if [ $# -ne 6 ]; then
    echo "Invalid number of parameters, you need to provide role name, region name, bucket name, prefix, express region name and express bucket name"
    echo "Usage: $0 S3RoleName us-west-2 s3torchconnector-test-bucket-name prefix-name/ us-east-1 s3torchconnectorclient-express-bucket-name"
    exit 1
fi

ROLE_NAME=$1
REGION_NAME=$2
BUCKET_NAME=$3
PREFIX=$4
EXPRESS_REGION_NAME=$5
EXPRESS_BUCKET_NAME=$6

FILE_NAME="tmp_cred.json"
TOKEN=`curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/${ROLE_NAME} >> ${FILE_NAME}
export AWS_ACCESS_KEY_ID=`cat ${FILE_NAME} | jq -r '.AccessKeyId'`
export AWS_SECRET_ACCESS_KEY=`cat ${FILE_NAME} | jq -r '.SecretAccessKey'`
export AWS_SESSION_TOKEN=`cat ${FILE_NAME} | jq -r '.Token'`
rm ${FILE_NAME}

export CI_REGION=${REGION_NAME}
export CI_BUCKET=${BUCKET_NAME}
export CI_PREFIX=${PREFIX}
export CI_EXPRESS_REGION=${EXPRESS_REGION_NAME}
export CI_EXPRESS_BUCKET=${EXPRESS_BUCKET_NAME}

cibuildwheel --output-dir wheelhouse --platform linux s3torchconnectorclient