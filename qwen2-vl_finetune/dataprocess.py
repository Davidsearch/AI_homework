import json

# 读取 JSON 文件
with open('annotations/val.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dic={}

for x in data['annotations']:

    d="train/VizWiz_val_"+str(x['image_id']).zfill(8)+".jpg,"+x["caption"]
    if  x['image_id'] not in dic:
        dic[ x['image_id'] ]=d
    else:
        dic[ x['image_id']] = max(dic[ x['image_id']], d, key=len)

with open("captions_val.txt", "w", encoding="utf-8") as file:
    for k,v in dic.items():
        file.write(v.replace("\n", "").replace("\r", "")+'\n')