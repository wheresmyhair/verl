sudo docker pull yushengsuthu/verl:verl-0.4.1_ubuntu-22.04_rocm6.3.4-numa-patch_vllm0.8.5_sglang0.4.6.post4

sudo docker tag yushengsuthu/verl:verl-0.4.1_ubuntu-22.04_rocm6.3.4-numa-patch_vllm0.8.5_sglang0.4.6.post4 verl-rocm:latest

cd $HOME

git clone https://github.com/wheresmyhair/verl.git && cd verl

git checkout amd-0.4.1

sudo docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  -e HOME=$HOME/eric \
  -e HF_HOME=$HOME/eric/hf_cache \
  --shm-size 128G \
  verl-rocm \
  /bin/bash