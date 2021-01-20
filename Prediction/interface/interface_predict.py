
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   interface_predict.py
@Time    :   2021/01/02 21:50:10
@Author  :   Hanlin Li 
@Version :   1.0
@Contact :   ahnz830@gmail.com
'''

# here put the import lib

from transformers import *
from torch import nn
import torch
from model import Model
from datetime import datetime

question = 'How many patients'
paragraph = '''In the present study, 118 adults undergoing first-time elective CABG or elective aortic valve replacement between August 2012 and July 2015 were randomly assigned by a computer program into two groups: the RAP group (n = 54) in which the retrograde autologous priming was applied and the non-RAP (n = 64) group in which the same setting was used without the possibility to save priming volume (Fig. 1). All patients were admitted to our unit the day before the planned operation. All patients received aspirin 100 mg/d until the day before the operation.Exclusion criteria were age < 18 years, LVEF ≤ 20%, emergency operations, reoperations, combined procedures, myocardial infarction 24 h before surgery, preoperative cortisone, coumarin, dual platelet inhibitor,or IV heparin therapy, thrombocytopenia (< 100 Gpt/l),liver disease, preoperative dialysis, hematological or oncological systemic disease or systemic infection.'''

MyBert = Model()
answering,score = MyBert.predict(question,paragraph)

print('Answer: ',answering)
nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("\n"+"=========="*8 + "%s"%nowtime)
print('\nStart_score is: {:.4f} \nEnd_score is: {:.4f}'.format(score['start_score'],score['end_score']))