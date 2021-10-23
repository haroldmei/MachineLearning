## Pay by seconds for your GPU 

(Advance cloud users)  
For people who are interested in those big modules(such as BERT, GPT, etc.), there is no better choices than utilising a cloud GPU from GCP or AWS for your research work. With just a couple of command lines, one can create a powerful Machine Learning computer to test one's idea out, without too much concern about setting up the computation environment. Cloud computing isn't supposed to be more expensive than physical servers, and should be much easier to access. However after numerous times making use of cloud GPUs, whether you are a student in a university or a recent AI graduate student, you often find yourself facing large amount of bills from either Google or Amazon or Micrisoft for your unawareness of excessive use of the cloud resource.    

I've been in such situations multiple times, and many times I have to pay with my own money, because I did use those resource with my own account. Thanks to my current employer I get to use the cloud resources with my employer's support; Yet I still find it useful to find out ways to minimize the the cost of my use of cloud resources.   

cloudrun.sh is a handy tool that can help me use cloud resource effectively, especially when I am in need of several cutting edge GPUs to test my ideas. The idea is straightforward: Develop locally, and train on the cloud. The resource used include only a cloud VM with a GPU, and the cloud storage.  

### Model development cycle
The steps for your model development (Ubuntu or Debian users only):     
* A basic home PC with 8G RAM and a NVIDIA GTX GPU is enough to start with. After your model code passes your local PC, it's time to bring up a cloud GPU to run your code.  
* Before bringing up the GPU VM with cloudrun.sh, some prerequisites are needed:
 1. you need a valid gcloud account, including a project id; and prepare these environment variables in system:   
   * $GCP_ACCOUNT: the google cloud account, your email address
   * $GCLOUD_PROJECT: the project id mentioned above; free tier or bill account
   * $GCS_DIR: yes, a google storage directory; you need a gcs location to store your data and models
   * $VM_INSTANCE: this is the virtual machine name you set, set it the first time and use it always
   * $VM_USER: this is the home user of the abouve VM.
 2. after all the above info are ready, put them in your linux .bashrc export section. (Windows users are similar)
 3. save cloudrun.sh to your local bin folder(~/bin), and >source ~/.bashrc
* Put all your model code, data and output model in the same folder, such as following. develop locally by using this folder structure. This means you will need to run 'python main.py' successfully   
  |- VPP
      |- data
      |- model
      |- src
      |- checkpoint
      |- main.py
* Train your model with a cloud GPU with 'cloudrun.sh main.py'. Of course this time set your 'BATCHSIZE', 'EPOCHS' and your embedding size a much larger value. :thump :thump   


