#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/01/02 15:03:50
@Author  :   Hanlin Li 
@Version :   2.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib
from torch import nn
import torch
from transformers import *

# config = AlbertConfig.from_json_file('path')
# path = "ktrapeznikov/albert-xlarge-v2-squad-v2"
path = 'bert-large-uncased-whole-word-masking-finetuned-squad'

def corrected_answer(answer):
        correcting_answer =''
        corrected_answer  =''                                   #Remove repeated punctuation
        for word in answer.split():
            if word != "'":
                if word[0] == '▁':
                    word = word[1:]
                    correcting_answer += ' ' + word
                else:
                    correcting_answer += ' ' + word         #für Albert und Roberta
        box = []                                    #divide a string in many strings and save in a list
        for word in correcting_answer.split():
            box.append(word)
        for i in range(len(box)):
            if i == len(box)-1:
                corrected_answer += ' '+ box[i]
            elif box[i] != box[i+1]:
                corrected_answer += ' '+ box[i]
        return corrected_answer


class Model:
    def __init__(self):
        super(Model, self).__init__()
        # self.tokenizer = AlbertTokenizer("path")
        self.tokenizer = BertTokenizer.from_pretrained(path)
        # QA_model = AlbertForQuestionAnswering('path')
        QA_model = BertForQuestionAnswering.from_pretrained(path)
        QA_model = QA_model.eval()
        self.model = QA_model
    
    # def corrected_answer(answer):
    #     correcting_answer =''
    #     corrected_answer  =''                                   #Remove repeated punctuation
    #     for word in answer.split():
    #         if word != "'":
    #             if word[0] == '▁':
    #                 word = word[1:]
    #                 correcting_answer += ' ' + word
    #             else:
    #                 correcting_answer += '' + word         #für Albert und Roberta
    #     box = []                                    #divide a string in many strings and save in a list
    #     for word in correcting_answer.split():
    #         box.append(word)
    #     for i in range(len(box)):
    #         if i == len(box)-1:
    #             corrected_answer += ' '+ box[i]
    #         elif box[i] != box[i+1]:
    #             corrected_answer += ' '+ box[i]
    #     return corrected_answer
    
    def predict(self,question,paragraph):
        encoded_text = self.tokenizer(
            text = question,
            text_pair = paragraph,
            truncation = True,
            padding = True,
            return_tensors = 'pt'
        )
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        token_type_ids = encoded_text['token_type_ids']
        
        with torch.no_grad():
            start_scores, end_scores = self.model(input_ids = input_ids,
                                                  attention_mask = attention_mask,
                                                  token_type_ids = token_type_ids)
        
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        if start_index >= end_index:
            answer= ' '.join(tokens[end_index:start_index+1])
        else:
            answer= ' '.join(tokens[start_index:end_index+1])
        
        answer = corrected_answer(answer)
        
        s_score = start_scores.squeeze()[start_index.data].data
        e_score = end_scores.squeeze()[end_index.data].data
        
        return (
            answer,
            dict(zip(['start_score','end_score'],[s_score,e_score]))
        )

model = Model()

def get_model():
    return model