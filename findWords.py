from asyncio import events
from itertools import tee
from logging import root
from logging.config import valid_ident
from tokenize import group
from typing import Dict, List
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from concurrent import futures


# def getGround(img, box):
#     x, y, w, h = box
#     whiteGround = 0
#     darkGround = 0

#     for r in range(x, x+w):
#         for c in range(y, y+h):
#             if img[c][r] == 255:
#                 darkGround += 1
#                 continue
#             whiteGround += 1
    
#     if whiteGround > darkGround:
#         return  whiteGround
#     return 0



# def getContrastColor(rbg):
#     return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

# def getContrastImg(img):
#     w, h =  len(img), len(img[0])
#     imgContrast = img.copy()

#     for c in range(w):
#         for r in range(h):
#             imgContrast[c][r] = getContrastColor(img[c][r])
    
#     return imgContrast



# st = time.time()

# img = cv2.imread('images/1.png')
# contrastImg = getContrastImg(img)
# cv2.imwrite('imageContrast.jpg', contrastImg)
# contrastImg = cv2.imread('imageContrast.jpg')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# contrastGrayImg = cv2.cvtColor(contrastImg, cv2.COLOR_BGR2GRAY)

# binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU 
#                                             | cv2.THRESH_BINARY_INV)[1]
# contrastBinaryImg = cv2.threshold(contrastGrayImg, 0, 255, cv2.THRESH_OTSU 
#                                             | cv2.THRESH_BINARY_INV)[1]

# cv2.imwrite('well.jpg', img[434: 434+20 , 462:462+63])
# cv2.imwrite('together.jpg', img[ 434: 434+20, 536:536+124])



# data0 = pytesseract.image_to_data(img, output_type=Output.DICT)
# data1 = pytesseract.image_to_data(contrastImg, output_type=Output.DICT)



# validWords0 = {}
# words0 = {}



# for i in range(len(data0['text'])):
#     if data0['text'][i] != '' and not data0['text'][i].isspace():
#         x, y, w, h = data0['left'][i], data0['top'][i], data0['width'][i], data0['height'][i]
#         words0[str((x, y, w, h))] = {'text': data0['text'][i], 
#                                 'ground': getGround(binaryImg, (x, y, w, h))}
    
# words1 = {}

# for i in range(len(data1['text'])):
#     if data1['text'][i] != '' and not data1['text'][i].isspace():
#         x, y, w, h = data1['left'][i], data1['top'][i], data1['width'][i], data1['height'][i]
#         words1[str((x, y, w, h))] = {'text': data1['text'][i], 
#                                     'ground': getGround(contrastBinaryImg, (x, y, w, h))}

#         print((x, y, w, h), ' : ', words1[str((x, y, w, h))])
# validWords = []


# for i in words0:
#     if words0[i]['ground'] > 0:
#         if (i in words1 
#             and words0[i]['ground'] < words1[i]['ground']):
#             validWords.append({'text':words1[i]['text'], 'box':eval(i)})
#             continue
#         validWords.append({'text':words0[i]['text'], 'box':eval(i)})


# for i in words1:
#     if words1[i]['ground'] > 0:
#         if (i in words0 
#             and words1[i]['ground'] < words0[i]['ground']):
#             validWords.append({'text':words0[i]['text'], 'box':eval(i)})
#             continue
#         validWords.append({'text':words1[i]['text'], 'box':eval(i)})

# validWords = sorted(validWords, key = lambda x: (x['box'][1], x['box'][0]))
# for i in validWords:
#     # print(i['box'], ': ', ': ', i['text'])
#     x, y, w, h = i['box']
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#     cv2.putText(img, i['text'], (x, y), 1, 1, (0, 255, 0))

# # cv2.rectangle(img, (462, 4), (x+w, y+h), (0, 255, 0), 1)
# # cv2.putText(img, i['text'], (x, y), 1, 1, (0, 255, 0))

# cv2.imwrite('output1.png', img)

# print('Excuse time: ', time.time() - st)

class A:
    def __init__(self, n) -> None:
        self.n = n

arr = [[2, 2], [1, 0], [2,1], [3,1], [4,1]]

arr = sorted(arr, key=lambda x: (x[0]*x[1]) )
print(arr)
l = len(arr)

for i in range(l):
    print(arr[i])
    for j in range(i+1, l):
        if j == 2:
            arr[j] = None
print(arr)

# for i in arr:
#     for j in arr:
#         if j.n==2:
#             arr.remove(j)