# -*- coding: utf-8 -*-
# @Author  : Qixin Li
# @Time    : 2021/5/27 上午11:00

from finetune.MUSE_dict.langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

if __name__=="__main__":
    traditional_sentence = 'uniform 軍服'
    simplified_sentence = Traditional2Simplified(traditional_sentence)
    fw = open("zh.txt","w")
    with open("tzh.txt","r") as f:
        for line in f.readlines():
            fw.write(Traditional2Simplified(line))
