# sequence-tagging
sequence tagging

### bert_model
从 https://huggingface.co/models 下载bert-base-chinese模型，解压在pretrained_models下

bert-base-chinese目录结构如下：
```
bert-base-chinese/
├── config.json
├── pytorch_model.bin
└── vocab.txt
```

### 训练10轮的评测结果
```
              precision    recall  f1-score   support

        game       0.78      0.84      0.81       295
     address       0.57      0.64      0.61       373
       movie       0.80      0.77      0.79       151
    position       0.76      0.82      0.79       433
        name       0.88      0.86      0.87       465
       scene       0.66      0.66      0.66       209
  government       0.74      0.84      0.79       247
organization       0.73      0.79      0.76       367
     company       0.75      0.82      0.78       378
        book       0.74      0.80      0.77       154

   micro avg       0.74      0.79      0.77      3072
   macro avg       0.75      0.79      0.77      3072
```