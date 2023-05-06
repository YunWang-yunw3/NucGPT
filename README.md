# NucGPT

## Purpose
Local Document-Specific Chat. Initially for UIUC NPRE. <img src=“images/block-I-icon-512x512w.png” alt=“block-I-icon” width=“10”/>
<!-- ![block-I-icon-512x512w](https://user-images.githubusercontent.com/73211705/235240284-590330de-7552-452d-8bbe-10c438d70f8d.png) -->

## Setup
### Download A Language Models (Only Works for RWKV atm)
- RWKV : https://huggingface.co/BlinkDL
- Llama : 
  - https://github.com/facebookresearch/llama 
  - https://www.reddit.com/r/LocalLLaMA/wiki/models/

### Clone Repo
```
git clone https://github.com/YunWang-yunw3/NucGPT.git
cd NucGPT
```
Then in this repository:
```
pip install -r requirements.txt
```

## Run
1. run parse.py if you do not have index and data yet
2. run main.py

## Dev Logs
### April 27th: Initial, generic HGTR text with 400M RWKV model. Looking into 4-bit quntized 7B and 13B models.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YunWang-yunw3/NucGPT&type=Date)](https://star-history.com/#YunWang-yunw3/NucGPT&Date)
