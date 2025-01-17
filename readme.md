## 实验准备
##  数据集
请到[https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)下载对应的训练集和验证集到`qwen2-vl_finetune`中的data文件夹中，
```
../qwen2-vl_finetune/data
├── captions.txt
├── captions_val.txt
├── train
├── test
└── val
```
如图中所示，其中的`captions.txt` `captions_val.txt`请先将annotations.zip文件夹中的文件解压后使用`qwen2-vl_finetune/dataprocess.py`调整对应参数后处理
(也就是调整对应数据集中图片数目，数据集名称)

## 环境准备
请自行按照`requirements.txt`配置conda环境，预计运行需要20G左右的显存。部分实验在Tesla P40显卡下需要训练20～30个小时

## Lora_train
我们第一个实验是关于大模型微调的实现，需要先到[hugging face](https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct)上下载通用义千问VL2b的模型文件夹。  

将其放在文件夹`qwen2-vl_finetune`下，运行

- `train.py` 即可使用LoRA微调Qwen2模型，模型会在微调结束后自动保存到模型文件夹。
- `Lora_without_Ada_runable.py` 这个是Lora微调的另一个实现和`train.py`类似
- `Lora_deepspeed.py`这个是使用了deepspeed下进行的实验。
- `test.py` 设置模型名称，即可使用对应模型对单张图片进行测试

训练完成之后，使用`val_bleu.py`进行评测，修改模型路径，即可加载不同的Qwen2模型进行bleu分数的评测

`run.py`与`run_with_trans.py`两个文件类似，会启动调整过后的Qwen模型作为服务器后端，接收前端服务。后者是自带翻译的版本，原本实现是通过VisionFive2开发板作为Client,目前因为技术原因所以暂时搁置。所以目前这两个后台服务并没有前端。

## BLIP_train
实验之后发现Qwen2作为后端，相应速度未能达到我们的设想，于是我们设计了一个新的实验。通过半监督学习训练BLIP模型，加快响应速度。
### 准备
这个实验前要求`qwen2-vl_fineture`文件夹下至少要有一个Qwen模型的文件夹，并且设置对应的文件夹名称。
### 第一轮训练
运行`blip/first_fine_tune_blip.py`（工作目录在blip下），即可使用train数据集对blip进行微调，最终会计算相应的bleu分数，以及效果图。
### 第二轮训练
第二轮训练需要Qwen2模型生成对应的训练集用于第二轮的训练，这个数据集的图像部分是test,而captions则是由Qwen2生成，运行`qwen2-vl_fine`文件夹下的`make_test.py`则生成`
`test.txt`。
确保这test文件夹和test.txt文件存在后，运行`second_fine_tune.py`,进行第二轮的训练。这也可以通过设置模型路径，从blip-base开始训练。
### 验证
通过`run_script.py`和`new_model_test.py`即可对两个模型进行测试。

## web
通过运行`web/run.py`即可在服务器下4321端口创建一个网页服务器，使用运行服务器的机器访问[http://localhost:4321/](http://localhost:4321/)即可运行客户端。
- 注意：如果这里用的不是Server上的浏览器，可能会因为浏览器安全特性导致无法访问摄像头，可以通过修改hosts文件的方式解决，即把Serverip 改为localhost即可。



## 失败的尝试
我们尝试了很多失败的方案，有些是训练时间的问题，有些是设计方案的问题，我们将其总结之后代码存放在`Rse_GPT2`文件夹下。
### RESNET—GPT2模型
我们设计了一个模型，即给GPT2添加识图功能，我们使用预训练的Resnet50处理数据，然后将Resnet50的输出传入GPT2,进行训练。这个模型最大的问题是我们使用了GPT2的预训练模型，导致Resnet50的输出错误的映射的到GPT2输入vocab输入空间中，在错误的训练60小时后我们发现了问题，随后放弃了这个方案。可以在`gr_train.py`中查看
### 蒸馏学习
受到deepseek的影响，我们认为可以用`Rse_GPT2`去学习Qwen的logit输出分布，可以更快的收敛，但是这个方案没有统一Qwen和GPT2对应的vocab，（其实是我们错误的认为不同vocab的相同的ids对应的词相同，实现了错误的`align_logits`。）这个错误的方案是可以在`kd_train.py`中查看。