source /home/dancer/anaconda3/bin/activate internvl

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8

export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000
export CONTROLLER_URL=http://0.0.0.0:$CONTROLLER_PORT
export SD_WORKER_URL=http://0.0.0.0:$SD_SERVER_PORT

# 4090系列禁用P2P和IB
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1