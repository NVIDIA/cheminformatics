MEGAMOLBART_MODEL_VERSION=0.1
MEGAMOLBART_MODEL_PATH=/models/megamolbart

DOWNLOAD_URL="https://api.ngc.nvidia.com/v2/models/nvidia/clara/megamolbart/versions/${MEGAMOLBART_MODEL_VERSION}/zip"
echo -e "${YELLOW}Downloading model megamolbart to ${MEGAMOLBART_MODEL_PATH}...${RESET}"

mkdir -p ${MEGAMOLBART_MODEL_PATH}
set -x
wget -q --show-progress \
    --content-disposition ${DOWNLOAD_URL} \
    -O ${MEGAMOLBART_MODEL_PATH}/megamolbart_${MEGAMOLBART_MODEL_VERSION}.zip
mkdir ${MEGAMOLBART_MODEL_PATH}
unzip -q ${MEGAMOLBART_MODEL_PATH}/megamolbart_${MEGAMOLBART_MODEL_VERSION}.zip \
    -d ${MEGAMOLBART_MODEL_PATH}
