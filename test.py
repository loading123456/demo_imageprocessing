import string
from asyncio import events
from tokenize import String, group
from typing import Dict, List
from xmlrpc.client import Boolean
from xxlimited import new
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from concurrent import futures


class Word:
    def __init__(self, box, text, confi) -> None:
        self.text = text
        self.startPoint = [box[0], box[1]]
        self.endPoint = [box[0] + box[2], box[1] + box[3]]
        self.center = [(box[0] + box[2]) / 2, (box[1] + box[3]) * 0.5]
        self.confident = int(confi)


class Line:
    def __init__(self, word:Word) -> None:
        self.words = [word]
        self.wordsSize = 1
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.center = word.center
        self.epsilon = (self.endPoint[1] - self.startPoint[1]) * 0.2
        self.spaceSize = (self.endPoint[1] - self.startPoint[1]) * 0.3 * 2.5

    def __updateLine(self, word:Word) -> None:
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        
        self.center[0] = (self.center[0] + word.center[0]) / 2
        self.center[1] = (self.center[1] + word.center[1]) / 2

        self.epsilon = (self.endPoint[1] - self.startPoint[1]) * 0.2
        self.spaceSize = (self.endPoint[1] - self.startPoint[1]) * 0.3 * 2.5
        self.wordsSize += 1

    def __getPosition(self, word:Word) -> int:
        if abs(self.center[1] - word.center[1]) <= self.epsilon:
            position = 0

            for node in self.words:
                if node.center[0] < word.center[0]:
                    position += 1
                else:
                    break
            if position == 0:
                distance = self.words[0].startPoint[0] - word.endPoint[0]
                if distance <= self.spaceSize:
                    return 0
                return -1

            if position == self.wordsSize:
                distance = word.startPoint[0] - self.words[-1].endPoint[0]
                if distance <= self.spaceSize :
                    return position
                return -1
            # print('Position: ', position)
            lastDistance =  word.startPoint[0] - self.words[position - 1].endPoint[0]
            nextDistance = self.words[position].startPoint[0] - word.endPoint[0]

            if (lastDistance <= self.spaceSize 
                and nextDistance <= self.spaceSize
            ):
                return position
        return -1

    def insertWord(self, word: Word) -> Boolean:
        position = self.__getPosition(word)
        
        if position != -1:
            if position == self.wordsSize:
                self.words.append(word)
            else:
                self.words.insert(position, word)
            self.__updateLine(word)
            return True
        return False

    def showText(self):
        text = self.getText()
        # print('size: ', len(self.words), self.startPoint, self.endPoint,  ' -- ',text)

    def getText(self):
        text = ''
        for i in range(len(self.words)):
            text += self.words[i].text + ' '
        return text
    def mergeText(self ,img):
        s = 0
        x, w = 0, 0 
        for word in self.words:
            if word.confident > 50:
                print("w: ", w)
                if w != 0:
                    position = self.words.index(word)
                    newWord = Word([x, self.startPoint[1], w - x, self.endPoint[1]-self.startPoint[1]],'', 50)
                    contrastImg = getContrastImg(img, x, self.startPoint[1], w - x, self.endPoint[1]-self.startPoint[1])
                    cv2.rectangle(img, newWord.startPoint, newWord.endPoint, (0, 255, 255), 2)
                    text = pytesseract.image_to_string(contrastImg)
                    print('contrast: ',text)
                    newWord.text = text
                    self.words.insert(position, newWord)
                    w = 0
                x = word.startPoint[0]
            else:
                print(word.text)
                w = int(word.endPoint[0])
                self.words.remove(word)
                self.wordsSize -= 1
            if w != 0:
                position = -1
                newWord = Word([x, self.startPoint[1], w - x, self.endPoint[1]-self.startPoint[1]],'', 50)
                contrastImg = getContrastImg(img, x, self.startPoint[1], w - x, self.endPoint[1]-self.startPoint[1])
                cv2.rectangle(img, newWord.startPoint, newWord.endPoint, (0, 255, 255), 1)
                # text = pytesseract.image_to_string(contrastImg)
                text = 'Detect'
                print('contrast: ',text)
                newWord.text = text
                self.words.insert(position, newWord)
                w = 0
        print("=========================================================")
            
class Lines:
    def __init__(self) -> None:
        self.lines = []
    
    def insertWord(self, word:Word):
        inserted = False
        for line in self.lines:
            if line.insertWord(word):
                inserted = True
                break
        
        if not inserted:
            self.__createLine(word)

    def __createLine(self, word):
        self.lines.append(Line(word))

    def show(self):
        for line in self.lines:
            line.showText()
            print("==================================================")
    
    def sort(self):
        self.lines = sorted(self.lines, key= lambda x: (x.center[1], x.center[0]))



class TextBox:
    def __init__(self, line:Line) -> None:
        self.lines = [line]
        self.center = line.center
        self.startPoint = line.startPoint
        self.endPoint = line.endPoint
        self.epsilon = (self.endPoint[1] - self.startPoint[1]) * 0.2 * 2

    def __getPosition(self, line:Line):
        if line.startPoint[1] - self.lines[-1].endPoint[1] <= self.epsilon * 3:
            if abs(line.startPoint[0] - self.lines[-1].startPoint[0]) <= self.epsilon * 8:
                return True
        return False
 
    def insertLine(self, line:Line):
        position = self.__getPosition(line)
        if position:
            self.lines.append(line)
            return True
        return False
    def show(self):
        for line in self.lines:
            line.showText()
        print("------------------------------------------------------------------")


class TexBoxs:
    def __init__(self) -> None:
        self.textboxs = []

    def insertLine(self, line: Line):
        inserted = False
        for textBox in self.textboxs:
            if textBox.insertLine(line):
                inserted = True
                break

        if not inserted:
            self.__createTextBox(line)

    def __createTextBox(self, line):
        self.textboxs.append(TextBox(line))

    def show(self):
        for textBox in self.textboxs:
            textBox.show()

    

def getContrastColor(rbg):
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img, x, y, w, h):
    # print(len(img), len(img[0]), x, y, h, w)
    # w, h =  len(img), len(img[0])
    imgContrast = img[y: y + h,x:x+w].copy()
    # print(len(imgContrast), len(imgContrast[0]))
    for c in range(h):
        for r in range(w):
            imgContrast[c][r] = getContrastColor(imgContrast[c][r])
    
    return imgContrast


def getValidWords(imgData, img):
    validWordsInImg = []
    dataLen = len(imgData['text'])

    for i in range(dataLen):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])
        # if (int(imgData['conf'][i]) < 50
        #     and (
        #         x != 0
        #         and y != 0
        #         and w != len(img[0])
        #         and h != len(img)
        #     )
        # ):
        #     contrastImg = getContrastImg(img, x, y, w, h)
        #     rawText = pytesseract.image_to_string(contrastImg)
        #     imgData['text'][i] = rawText.replace(' ','')

        if(
             imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
        ):
            validWordsInImg.append([imgData['text'][i],
                                (x, y, w, h),
                                imgData['conf'][i] ]),
                                
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
            cv2.putText(img, imgData['text'][i], (x, y), 1, 1, (0, 255, 0))
                
            # print((x, y, w, h),len(imgData['text'][i]), ' | ', imgData['text'][i])
        # elif int(imgData['conf'][i]) > 0 and imgData['text'][i] != '' and not imgData['text'][i].isspace():
        #     if getGround(binaryImg, (x, y, w, h)):
        #         validWordsInImg.append([imgData['text'][i],
        #             imgData['conf'][i], 
        #             (x, y, w, h)])

    return validWordsInImg

def func(imgPath):


    img = cv2.imread('images/' + imgPath)
    # contrastImg = getContrastImg(img)
    # cv2.imwrite('cache/' + imgPath +'.jpg', contrastImg)
    # contrastImg = cv2.imread('cache/'+imgPath + '.jpg')

    # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU 
    #                                         | cv2.THRESH_BINARY_INV)[1]
    st = time.time()

    imgData = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
    # contrastImgData = pytesseract.image_to_data(contrastImg, lang='eng', output_type=Output.DICT)
    # print(pytesseract.image_to_data(img, lang='eng'))

    wordsInImg = getValidWords(imgData, img)

    print(len(wordsInImg))


    wordsInImg = sorted(wordsInImg, key = lambda x: (x[1][0], x[1][1]))
    lines = Lines()
    for value in wordsInImg:
        if value[1][2] < len(img) and value[1][3] < len(img[0]):
            word = Word(value[1], value[0], value[2])
            lines.insertWord(word)
            # cv2.rectangle(img, [value[1][0], value[1][1]], [value[1][0] + value[1][2], value[1][1] + value[1][3]], (255, 0, 255), 1)

    # lines.show()
    lines.sort()

    # for line in lines.lines:
    #     line.mergeText(img)

    textboxs = TexBoxs()

    for line in lines.lines:
        textboxs.insertLine(line)

    textboxs.show()

    # for item  in lines.lines:
    #     cv2.rectangle(img, item.startPoint, item.endPoint, (255, 0, 255), 1)
    #     cv2.putText(img, item.getText(), item.startPoint, 1, 1, (0, 255, 0))

    # for word in wordsInImg:
    #     print(word)

    # for item  in wordsInContrastImg:
    #     x, y, w, h = item[2]
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (55, 0, 255), 2)



    cv2.imwrite('output/20_' + imgPath, img)
    print("Excuse time: ", time.time() - st)



# func('1.png')
# func('2.png')
# func('3.png')
# func('4.png')
func('5.png')
# func('a5.png')
# func('a6.png')