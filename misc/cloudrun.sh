if [[ -z "${GCP_ACCOUNT}" ]]; then
  echo  "GCP_ACCOUNT undefined, add 'export GCP_ACCOUNT=xxx@gmail.com' in .bashrc and source it"
  exit 1
fi

if [[ -z "${GCLOUD_PROJECT}" ]]; then
  echo  "GCLOUD_PROJECT undefined, add 'export GCLOUD_PROJECT=project_id' in .bashrc and source it"
  exit 1
fi

if [[ -z "${GCS_DIR}" ]]; then
  echo  "GCS_DIR undefined, add 'export GCS_DIR=bucket_url' in .bashrc and source it"
  exit 1
fi

if [[ -z "${VM_INSTANCE}" ]]; then
  echo  "VM_INSTANCE undefined, add 'export VM_INSTANCE=vmname' in .bashrc and source it"
  exit 1
fi

if [[ -z "${VM_USER}" ]]; then
  echo  "VM_USER undefined, add 'export VM_USER=user' in .bashrc and source it"
  exit 1
fi

# A100: --machine-type a2-highgpu-1g
# T4:   --machine-type n1-standard-4 --accelerator type=nvidia-tesla-t4,count=1 \
# gcloud compute instances create $VM_INSTANCE \
#     --network $GCP_NETWORK --subnet $GCP_SUBNET --no-address \
#     --zone us-central1-a \
#     --boot-disk-type pd-standard \
#     --boot-disk-size 200GB \
#     --metadata install-nvidia-driver=True,proxy-mode=project_editors,enable-oslogin=TRUE \
#     --machine-type a2-highgpu-1g \
#     --image-family pytorch-latest-gpu-ubuntu-1804 \
#     --image-project deeplearning-platform-release \
#     --maintenance-policy TERMINATE --restart-on-failure \
#     --preemptible

# gcloud compute instances add-tags $VM_INSTANCE --tags=iap-enabled

gcloud compute instances start $VM_INSTANCE --zone us-central1-a

STATUS=$(gcloud compute ssh $VM_INSTANCE --zone=us-central1-a --ssh-flag="-q" --command 'echo 123')
while [[ -z "$STATUS" ]]
do
    STATUS=$(gcloud compute ssh $VM_INSTANCE --zone=us-central1-a --ssh-flag="-q" --command 'echo 123')
    sleep 5; 
done

WORKSPACE=$(echo $PWD | rev | cut -d'/' -f 1 | rev)
GCSWORKSPACE=$GCS_DIR/$WORKSPACE


# gcloud compute ssh --zone us-central1-a $VM_USER@$VM_INSTANCE --ssh-flag="-L 8080:localhost:8080"
# gcloud compute ssh --zone us-central1-a $VM_USER@$VM_INSTANCE -- pip3 -r -s < requirements.txt

# copy to gcs
echo "sync $WORKSPACE to $GCSWORKSPACE"
gsutil -m rsync -r . $GCSWORKSPACE

# get current active account; 
ACTIVE_ACCOUNT=$(gcloud compute ssh --zone us-central1-a $VM_USER@$VM_INSTANCE -- 'gcloud config list account --format "value(core.account)"')

# login, copy data, run, upload
if echo $ACTIVE_ACCOUNT | grep $GCP_ACCOUNT; then
    echo "auth skipped"
else
    echo "auth needed, install packages as well"
    gcloud compute ssh --zone us-central1-a $VM_USER@$VM_INSTANCE -- 'gcloud auth login' 
fi

# && pip3 install transformers tensorboard \

COMMAND="gcloud config set project $GCLOUD_PROJECT 
    && mkdir -p $WORKSPACE \
    && gsutil -m rsync -r $GCSWORKSPACE $WORKSPACE \
    && cd $WORKSPACE \
    && /opt/conda/bin/pip3 install -r requirements.txt \
    && /opt/conda/bin/python3 $1 \
    && gsutil -m rsync -r . $GCSWORKSPACE"

echo !! Command: $COMMAND

gcloud compute ssh --zone us-central1-a $VM_USER@$VM_INSTANCE -- $COMMAND

# gsutil -m rsync -r $GCSWORKSPACE .

gcloud compute instances stop $VM_INSTANCE --zone us-central1-a

# gcloud compute instances delete $VM_INSTANCE --zone us-central1-a