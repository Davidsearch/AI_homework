实验准备
- 数据集
请到[https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)下载对应的训练集和验证集到`qwen2-vl_finetune`中的data文件夹中，
```
../qwen2-vl_finetune/data
├── captions.txt
├── captions_val.txt
├── train
└── val
```
如图中所示，其中的`captions.txt` `captions_val.txt`请先将annotations.zip文件夹中的文件解压后使用`qwen2-vl_finetune/dataprocess.py`调整对应参数后处理
(也就是调整对应数据集中图片数目，数据集名称)

- Lora_train
我们第一个实验是关于大模型微调的实现，需要先到[hugging face](https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct)上下载通用义千问VL2b的那个模型，
