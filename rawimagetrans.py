from cgitb import text
from curses.ascii import isspace
from operator import mod
from typing import List
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from pytesseract import Output
import re
from PIL import Image, ImageFont, ImageDraw


def toWords(imgData, imgSize) -> List:
    words = []
    for i in range(len(imgData['text'])):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])
        if(imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
            and w < imgSize[0]
            and h < imgSize[1]
        ): 
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', imgData['text'][i])
            words.append(Word([x, y, w, h], imgData['conf'][i], text))
    return words


class Word:
    def __init__(self, box, confi, text) -> None:
        self.text = text
        self.startPoint = [box[0], box[1]]
        self.endPoint = [box[0] + box[2], box[1] + box[3]]
        self.centerPoint = [box[0] + box[2] * 0.5, box[1] + box[3] * 0.5]
        self.width = box[2]
        self.height = box[3]
        self.confident = int(confi)



def getContrastColor(rbg):
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img):
    contrastImg = img.copy()
    for y in range(len(contrastImg)):
        for x in range(len(contrastImg[y])):
            contrastImg[y][x] = getContrastColor(contrastImg[y][x])
    return contrastImg            


class UnArea:
    def __init__(self, word:Word) -> None:
        self.centerPoint = word.centerPoint
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.X_NOUN = 0.3
        self.Y_NOUN = 0.3
        self.height = self.endPoint[1] - self.startPoint[1]
        self.epsilonX = self.height * self.X_NOUN
        self.epsilonY = self.height * self.Y_NOUN
        self.text = ''
        self.size = 1


    def __getPosition(self, word:Word) -> int:
        if abs(self.centerPoint[1] - word.centerPoint[1]) <= self.epsilonY:
            if abs(word.startPoint[0] - self.endPoint[0]) <= self.epsilonX:
                return 1
        return -1

    def __updateLine(self, word:Word):
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        
        self.centerPoint[0] = (self.startPoint[0] + self.endPoint[0])/2
        self.centerPoint[1] =  (self.startPoint[1] + self.endPoint[1])/2
        self.epsilonX = (self.endPoint[1] - self.startPoint[1]) * self.X_NOUN
        self.epsilonY = (self.endPoint[1] - self.startPoint[1]) * self.Y_NOUN
        self.size += 1

    def insertArea(self, word:Word) -> Boolean:
        position = self.__getPosition(word)

        if position != -1:
            self.__updateLine(word)
            return True
        return False


    def __recognizeText(self, img):
        startX = int((self.startPoint[0] - self.epsilonX)
                        if self.startPoint[0] - self.epsilonX >= 0 
                        else self.startPoint[0])
        startY = int((self.startPoint[1] - self.epsilonX) 
                        if self.startPoint[1] - self.epsilonX >= 0 
                        else self.startPoint[1])

        endX = int((self.endPoint[0] + self.epsilonX) 
                        if self.endPoint[0] + self.epsilonX <= img.shape[0]
                        else self.endPoint[0]) 
        endY = int((self.endPoint[1] + self.epsilonX) 
                        if self.endPoint[1] + self.epsilonX <= img.shape[1]
                        else self.endPoint[1])

        contrastImg = getContrastImg(img[startY:endY, startX:endX])
        text = pytesseract.image_to_string(contrastImg, lang='eng')
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
        return text


    def toWord(self, img):
        return Word([self.startPoint[0], self.startPoint[1], 
                        self.endPoint[0] - self.startPoint[0],
                        self.endPoint[1] - self.startPoint[1]],
                     50,
                     self.__recognizeText(img))



class Line:
    def __init__(self, word:Word) -> None:
        self.words = [word]
        self.centerPoint = word.centerPoint
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.height = word.height
        self.width = word.width
        self.X_NOUN = 0.7
        self.Y_NOUN = 0.3
        self.epsilonX = self.height * self.X_NOUN
        self.epsilonY = self.height * self.Y_NOUN
        self.size = 1


    def __getPosition(self, word:Word) -> int:
        if abs(self.centerPoint[1] - word.centerPoint[1]) <= self.epsilonY:
            position = 0

            for node in self.words:
                if node.centerPoint[0] < word.centerPoint[0]:
                    position += 1
                else:
                    break
            if position == 0:
                distance = self.words[0].startPoint[0] - word.endPoint[0]
                if distance <= self.epsilonX:
                    return 0
                return -1

            if position == self.size:
                distance = word.startPoint[0] - self.words[-1].endPoint[0]
                if distance <= self.epsilonX :
                    return position
                return -1
            lastDistance =  word.startPoint[0] - self.words[position - 1].endPoint[0]
            nextDistance = self.words[position].startPoint[0] - word.endPoint[0]

            if (lastDistance <= self.epsilonX 
                and nextDistance <= self.epsilonX
            ):
                return position
        return -1


    def __updateLine(self, word:Word):
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        
        self.centerPoint[0] = (self.startPoint[0] + self.endPoint[0])/2
        self.centerPoint[1] = (self.startPoint[1] + self.endPoint[1])/2
        self.epsilonX = (self.endPoint[1] - self.startPoint[1]) * self.X_NOUN
        self.epsilonY = (self.endPoint[1] - self.startPoint[1]) * self.Y_NOUN
        self.width = self.endPoint[0] - self.startPoint[0]
        self.height = self.endPoint[1] - self.startPoint[1]
        self.size += 1

    def insertWord(self, word:Word) -> Boolean:
        position = self.__getPosition(word)
        if position != -1:
            if position == self.size:
                self.words.append(word)
            else:
                self.words.insert(position, word)
            self.__updateLine(word)
            return True
        return False

    def getBox(self):
        x = int((self.startPoint[0]))
        y = int(self.startPoint[1] - self.height * 0.1)
        w = int(self.width)
        h = int(self.height + self.height * 0.2)
        return x, y, w, h

    def getText(self):
        text = ''
        for word in self.words:
            text += word.text + ' '
        if text[0] == ' ':
            text = text[1:]
        return re.sub(' +', ' ', text)



class TextBox:
    def __init__(self, line:Line) -> None:
        self.lines = [line]
        self.lineSize = 1
        self.height = line.height
        self.epsilonHeight = self.height * 0.3
        self.lineSpace = self.height 
        self.textLines = []

    def __getPosition(self, line:Line):
        if (line.startPoint[1] - self.lines[-1].endPoint[1]  <=  self.lineSpace
                and line.startPoint[1] - self.lines[-1].endPoint[1] > 0
        ):
            if ((abs(self.lines[-1].startPoint[0] - line.startPoint[0]) 
                    <= line.epsilonX * 5)
                or (abs(line.endPoint[0] - self.lines[-1].endPoint[0]) 
                        <= line.epsilonX * 5)
                or (abs(line.centerPoint[0] - self.lines[-1].centerPoint[0])
                        <= line.epsilonX * 5)
            ):
                return 1
        return -1
 
    def insertLine(self, line:Line):
        position = self.__getPosition(line)
        if position != -1:
            self.lines.append(line)
            self.lineSize += 1
            return True
        return False
    
    def __translateText(self):
        text = ''
        for line in self.lines:
            self.textLines.append(line.getText())
            text += line.getText() + ' __1 ' 
        
        tText = (googletrans.Translator()
                    .translate(text, dest='vi').text)
        tText = re.sub('\\s+', ' ', tText).strip()
        self.textLines = re.split('__', tText)

        for i in range(len(self.textLines)):
            if len(self.textLines[i]) > 1 and self.textLines[i][0] == '1' :
                self.textLines[i] = self.textLines[i][1:]

    def draw(self, img):
        self.__translateText()
        for i in range(self.lineSize):
            if (not self.textLines[i].isspace() 
                and self.textLines[i] != ''
                and self.textLines[i].replace(" ", "") != self.lines[i].getText().replace(" ", "")
            ):
                x, y, w, h = self.lines[i].getBox()
                font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf', int(h))
                textWidth, textHeight = font.getsize(self.textLines[i])
                if textWidth < w:
                    textWidth = w
                textBox = Image.new(mode="RGBA", size=(textWidth, int(textHeight*2) ), color=(235, 235, 235))
                d = ImageDraw.Draw(textBox)
                d.text((0, 0), self.textLines[i], font=font, fill=(0, 0, 0))
                textBox.thumbnail((w, 1000  ), Image.ANTIALIAS)
                textBox = textBox.crop((0, 0, w, h))

                img.paste(textBox, (x, y), textBox.convert("RGBA"))




def translate(imgPath, savePath):
    st = time.time()
    img = cv2.imread(imgPath)
    imgData = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
    words = toWords(imgData, img.shape)
    words = sorted(words, key = lambda key: (key.startPoint[0]))
    unAreas = []

# Re recognize text 
    for word in words:
        if word.confident < 50:
            inserted = False
            for unArea in unAreas:
                if unArea.insertArea(word):
                    inserted = True
                    break
            if not inserted:
                unAreas.append(UnArea(word))

    for unArea in unAreas:
        words.append(unArea.toWord(img))

# Merge words to lines
    words = sorted(words, key = lambda key: (key.startPoint[0]))
    lines = []

    for word in words:
        if word.confident >= 50:
            inserted = False
            for line in lines:
                if line.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                lines.append(Line(word))


# Merge lines to text boxs
    lines = sorted(lines, key= lambda key: key.centerPoint[1])
    textBoxs = []

    for line in lines:
            inserted = False
            for textBox in textBoxs:
                if textBox.insertLine(line):
                    inserted = True
                    break 
            if not inserted:
                textBoxs.append(TextBox(line))
# Translate and Draw 

    outputImg = Image.open(imgPath).convert("RGB")

    for textBox in textBoxs:
        textBox.draw(outputImg)



    outputImg.save(savePath)


    print("Excuse time: ", time.time() - st)

translate(imgPath='images/1.png', savePath='output/a_1.png')
# translate(imgPath='images/2.png', savePath='output/a_2.png')
# translate(imgPath='images/3.png', savePath='output/a_3.png')
# translate(imgPath='images/4.png', savePath='output/a_4.png')
# translate(imgPath='images/5.png', savePath='output/a_5.png')
# translate(imgPath='images/6.png', savePath='output/a_6.png')
# translate(imgPath='images/7.png', savePath='output/a_7.png')
# translate(imgPath='images/8.png', savePath='output/a_8.png')
# translate(imgPath='images/9.png', savePath='output/a_9.png')
# translate(imgPath='images/10.jpg', savePath='output/a_10.png')
# translate(imgPath='images/14.png', savePath='output/a_14.png')
# translate(imgPath='images/15.png', savePath='output/a_15.png')