## 实验准备
##  数据集
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

## 环境准备
请自行按照`requirements.txt`配置conda环境，预计运行需要20G左右的显存。部分实验在Tesla P40显卡下需要训练20～30个小时

## Lora_train
我们第一个实验是关于大模型微调的实现，需要先到[hugging face](https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct)上下载通用义千问VL2b的那个模型，

将其放在文件夹`qwen2-vl_finetune`下，运行

- `train.py` 即可使用LoRA微调Qwen2模型，模型会在微调结束后自动保存
- `Lora_without_Ada_runable.py` 这个是Lora微调的另一个实现和`train.py`类似
- ``
- `test.py` 设置模型名称，即可使用对应模型对单张图片进行测试

训练完成之后，使用`val_bleu.py`进行评测，修改模型路径，即可加载不同的Qwen2模型进行bleu分数的评测

`run.py`与`run_with_trans.py`两个文件类似，会启动调整过后的Qwen模型作为服务器后端，接收前端服务。后者是自带翻译的版本，原本实现是通过VisionFive2开发板作为Client,目前因为技术原因所以暂时搁置。所以目前这两个后台服务并没有前端。

## Blip_train
实验之后发现Qwen2作为后端，相应速度未能达到我们的设想，于是我们设计了一个新的实验。通过半监督学习训练BLIP模型，加快响应速度。
### 准备
这个实验前要求`qwen2-vl_fineture`文件夹下至少要有一个Qwen模型的文件夹，