#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import spacy
import pkuseg

seg = pkuseg.pkuseg()           # 以默认配置加载模型
text = seg.cut('我爱北京天安门')  # 进行分词
print(text)

# Load English tokenizer, tagger, parser, NER and word vectors
nlp_en = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp_en(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# Load Chinese model
nlp_zh = spacy.load("zh_core_web_sm")

text = ("卷积神经网络，也就是convolutional neural networks （简称CNN），现在已经被用来应用于各个领域，物体分割啦，风格转换啦，自动上色啦blahblah，但是！！CNN真正能做的，只是起到一个特征提取器的作用！所以这些应用，都是建立在CNN对图像进行特征提取的基础上进行的。")
doc = nlp_zh(text)
print([(w.text, w.pos_) for w in doc])