#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   api.py
@Time    :   2021/01/02 15:03:33
@Author  :   Hanlin Li 
@Version :   1.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib
from fastapi import FastAPI,Depends
from pydantic import BaseModel
from typing import Dict

from .Q_A.model import Model,get_model

app = FastAPI()                                         # 创造实例

class Question_and_Answering_Request(BaseModel):        # 我们需要请求和回应两个body
    question : str
    paragraph: str

class Question_and_Answering_Response(BaseModel):
    answering: str
    score: Dict[str, float]


@app.post("/predict", response_model = Question_and_Answering_Response)

def predict(request:Question_and_Answering_Request, model: Model = Depends(get_model)):
    answering,score = model.predict(request.question,request.paragraph)
    
    
    return Question_and_Answering_Response(
        answering = answering,
        score=score,
    )