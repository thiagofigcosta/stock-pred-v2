#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils
from Dataset import Dataset


a=Dataset.Entry()
a.backward_values=[tuple([0])]
a.forward_values=[tuple([1])]

a.next=Dataset.Entry()
a.next.backward_values=[tuple([2])]
a.next.forward_values=[tuple([3])]

a.next.next=Dataset.Entry()
a.next.next.backward_values=[tuple([4])]
a.next.next.forward_values=[tuple([5])]

a.next.next.next=Dataset.Entry()
a.next.next.next.backward_values=[tuple([4])]
a.next.next.next.forward_values=None

a.regenerateIndexes()
print('size',a.getSize())
print('idx',a.getIndexArray(0))
print('idx-1',a.getIndexArray(1))
print('date',a.getDateArray(0))
print('val',a.getValueArray(0))
print('val-future',a.getValueArray(1))
print('index of nth',a.getNthPointer(2).index)
print('index of atIdx',a.getPointerAtIndex(2).index)
print('index of last',a.getPointerAtIndex(a.getSize()-1).index)
print('Next 6', Utils.getStrNextNWorkDays('06/06/2021',6))


print('------------------------\n')
a=Dataset.Entry(date_index='06/06/2021',value=1)
a.next=Dataset.Entry(date_index='07/06/2021',value=2)
a.next.next=Dataset.Entry(date_index='08/06/2021',value=3)
a.next.next.next=Dataset.Entry(date_index='09/06/2021',value=4)
a.next.next.next.next=Dataset.Entry(date_index='10/06/2021',value=5)
a.next.next.next.next.next=Dataset.Entry(date_index='11/06/2021',value=6)
a.next.next.next.next.next.next=Dataset.Entry(date_index='14/06/2021',value=7)
a.next.next.next.next.next.next.next=Dataset.Entry(date_index='15/06/2021',value=8)
a.regenerateIndexes()
print('indexes',a.getIndexArray())

print('Original:')
a.printRawDatesAndValues()

print('\nback=4, for=3:')
b=a.generateBackAndForward(4,3)
b.printRawDatesAndValues()
print('indexes',b.getIndexArray())
print('dates',b.getDateArray(consider_null_forwards=True))
print('values',b.getValueArray())

print('\nback=4, for=2:')
c=a.generateBackAndForward(4,2)
c.printRawDatesAndValues()
print('indexes',c.getIndexArray())

print('------------------------\n')
b=Dataset.Entry(date_index='06/06/2021',value='a')
b.next=Dataset.Entry(date_index='07/06/2021',value='b')
b.next.next=Dataset.Entry(date_index='08/06/2021',value='c')
b.next.next.next=Dataset.Entry(date_index='09/06/2021',value='d')
b.next.next.next.next=Dataset.Entry(date_index='10/06/2021',value='e')
b.next.next.next.next.next=Dataset.Entry(date_index='11/06/2021',value='f')
b.next.next.next.next.next.next=Dataset.Entry(date_index='14/06/2021',value='g')

a.mergeWith(b)

a.printRawDatesAndValues()