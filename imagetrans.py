from ast import Str
from operator import le
import string
from tokenize import String
from typing import Any, List, Tuple
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import time
import concurrent.futures
import re
from pytesseract import Output
from PIL import Image, ImageFont, ImageDraw
import googletrans 
import math 


class Box:
    def __init__(self, rect, bgColor ) -> None:
        self.bgColor = bgColor
        self.stPoint = [rect[0], rect[1]]
        self.enPoint = [rect[0] + rect[2], rect[1] + rect[3]]
        self.cePoint = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
        self.w = rect[2]
        self.h = rect[3]


class BoxLv1(Box):
    def __init__(self,box:Box) -> None:
        rect = box.stPoint + [box.w, box.h]
        super().__init__(rect, box.bgColor)
        self.X_NOUN = 1
        self.Y_NOUN = 0.3
        self.H_NOUN = 0.4
        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN

    def insertBox(self, box:Box) -> Boolean:
        if self.__isValid(box):
            self.__update(box)
            return True
        return False

    def __isValid(self, box:Box) -> Boolean:
        if(box.bgColor == self.bgColor
            and abs(self.enPoint[0] - box.stPoint[0]) <= self.epX
            and abs(self.cePoint[1] - box.cePoint[1]) <= self.epY
            and abs(self.h - box.h) <= self.epH
        ):
            return True
        return False

    def __update(self, box:Box) -> None:
        if self.stPoint[0] > box.stPoint[0]:
            self.stPoint[0] = box.stPoint[0]
        if self.stPoint[1] > box.stPoint[1]:
            self.stPoint[1] = box.stPoint[1]

        if self.enPoint[1] < box.enPoint[1]:
            self.enPoint[1] = box.enPoint[1]
        if self.enPoint[0] < box.enPoint[0]:
            self.enPoint[0] = box.enPoint[0]
        
        self.cePoint[0] = (self.stPoint[0] + self.enPoint[0])/2
        self.cePoint[1] =  (self.stPoint[1] + self.enPoint[1])/2

        self.h = self.enPoint[1] - self.stPoint[1]
        self.w = self.enPoint[0] - self.stPoint[0]

        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN




class BoxLv2(Box):
    def __init__(self, boxLv1:BoxLv1) -> None:
        rect = boxLv1.stPoint + [boxLv1.w, boxLv1.h]
        super().__init__(rect, boxLv1.bgColor)
        self.X_NOUN = 0
        self.Y_NOUN = 0
        self.H_NOUN = 0
        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN
        self.imagePath = ''
    
    def insertBox(self, boxLv1:BoxLv1)-> Boolean:
        if self.__isValid(boxLv1):
            self.__update(boxLv1)
            return True
        return False
    
    def __isValid(self, boxLv1:BoxLv1) -> Boolean:
        if(boxLv1.bgColor == self.bgColor
            and abs(self.enPoint[1] - boxLv1.stPoint[1]) <= self.epY
            and abs(self.h - boxLv1.h) <= self.epH
            and (
                    abs(self.stPoint[0] - boxLv1.stPoint[0]) < self.epX
                    or abs(self.cePoint[0] - boxLv1.cePoint[0]) < self.epX
                    or abs(self.enPoint[0] - boxLv1.enPoint[0]) < self.epX
                )
        ):
            return True
        return False

    def __update(self, boxLv1:BoxLv1) -> None:
        if self.stPoint[0] > boxLv1.stPoint[0]:
            self.stPoint[0] = boxLv1.stPoint[0]
        if self.stPoint[1] > boxLv1.stPoint[1]:
            self.stPoint[1] = boxLv1.stPoint[1]

        if self.enPoint[1] < boxLv1.enPoint[1]:
            self.enPoint[1] = boxLv1.enPoint[1]
        if self.enPoint[0] < boxLv1.enPoint[0]:
            self.enPoint[0] = boxLv1.enPoint[0]
        
        self.cePoint[0] = (self.stPoint[0] + self.enPoint[0])/2
        self.cePoint[1] =  (self.stPoint[1] + self.enPoint[1])/2

        if self.h < boxLv1.h:
            self.h = boxLv1.h
        self.w = self.enPoint[0] - self.stPoint[0]

        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN

    def saveToJpg(self):
        self.imagePath = 'cache/' + str(self.stPoint + self.enPoint) + '.jpg'
        x, y, w, h = self.getRect()
        im = None
        if self.bgColor == 0:
            im = img[y:h, x:w]
        else:
            im = getContrastImg(img[y:h, x:w])
        cv2.imwrite(self.imagePath, im)


    def getRect(self, type=0) -> List:
        x = self.stPoint[0]
        y = int(self.stPoint[1] - (self.epY/2 
                                    if(self.stPoint[1] - self.epY/2 >= 0) 
                                    else 0))
        endX = self.enPoint[0]
        endY = int(self.enPoint[1] + (self.epY/2 
                                    if(self.enPoint[1] - self.epY/2 <= img.shape[1]) 
                                    else 0))

        return [x, y, endX, endY]

class Word(Box):
    def __init__(self, rect, bgColor, text, confident ) -> None:
        super().__init__(rect, bgColor)
        self.text = text
        self.confident = confident
    
    def getRect(self) -> List:
        return self.stPoint + [self.w, self.h]

class Line(Box):
    def __init__(self, word:Word) -> None:
        super().__init__(word.getRect(), word.bgColor)
        self.text = word.text
        self.words = [word]
        self.X_NOUN = 1
        self.Y_NOUN = 0.4
        self.H_NOUN = 0.6
        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN
        self.amount = 1

    def insertWord(self, word:Word) -> Boolean:
        if self.__isValid(word):
            self.__update(word)
            return True
        return False

    def __isValid(self, word:Word) -> Boolean:
        if (abs(self.cePoint[1] - word.cePoint[1]) <= self.epY
            and abs(self.h - word.h) <= self.epH
            and word.stPoint[0] - self.enPoint[0] <= self.epX
        ):
            return True
        return False
    
    def __update(self, word:Word) -> None:
        if self.stPoint[0] > word.stPoint[0]:
            self.stPoint[0] = word.stPoint[0]
        if self.stPoint[1] > word.stPoint[1]:
            self.stPoint[1] = word.stPoint[1]

        if self.enPoint[0] < word.enPoint[0]:
            self.enPoint[0] = word.enPoint[0]
        if self.enPoint[1] < word.enPoint[1]:
            self.enPoint[1] = word.enPoint[1]
        
        self.cePoint[0] = (self.stPoint[0] + self.enPoint[0])/2
        self.cePoint[1] = (self.stPoint[1] + self.enPoint[1])/2
        self.w = self.enPoint[0] - self.stPoint[0]
        self.h = self.enPoint[1] - self.stPoint[1]

        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN
        self.text += ' ' + word.text

        self.amount += 1


    def getRect(self) -> List:
        x = self.stPoint[0]
        y = int(self.stPoint[1] - (self.epY/2 
                                    if(self.stPoint[1] - self.epY/2 >= 0) 
                                    else 0))
        endX = self.w
        endY = self.h + int(self.epY/1.5)

        return [x, y, endX, endY]


class Paragraph:
    def __init__(self, line:Line) -> None:
        self.lines = [line]
        self.avH = line.h
        self.X_NOUN = 0
        self.Y_NOUN = 0
        self.H_NOUN = 0.3
        self.epX = 0
        self.epY = line.h
        self.epH = self.avH * self.H_NOUN
        self.amount = 1
        self.al = [0, 0, 0]

    def insertLine(self, line:Line) -> Boolean:
        if self.__isValid(line):
            self.lines.append(line)
            self.amount += 1
            return True
        return False

    def __isValid(self, line:Line) -> Boolean:
        if (abs(self.lines[-1].enPoint[1] - line.enPoint[1]) <= self.epY
            and abs(self.avH - line.h) <= self.epH
        ):
            if (abs(self.lines[-1].stPoint[0] - line.stPoint[0]) <= self.epX
                and self.al[0] + 1 >= self.al[1]
                and self.al[0] + 1>= self.al[2]
            ):  
                self.al[0] += 1
                return True

            if (abs(self.lines[-1].cePoint[0] - line.cePoint[0]) <= self.epX
                and self.al[1] + 1 >= self.al[0]
                and self.al[1] + 1>= self.al[2]
            ): 
                self.al[1] += 1
                return True

            if (abs(self.lines[-1].cePoint[0] - line.cePoint[0]) <= self.epX
                and self.al[2] + 1 >= self.al[1]
                and self.al[2] + 1>= self.al[2]
            ): 
                self.al[2] += 1
                return True
        return False
                


    def draw(self, img):
        for i in range(self.amount):
            if (not self.textLines[i].isspace() 
                and self.textLines[i] != ''
                and self.textLines[i].replace(" ", "") != self.lines[i].getText().replace(" ", "")
            ):
                x, y, w, h = self.lines[i].getRect()
                bg = Image.new(mode="RGBA", size=(w, h), color=(235, 235, 235))
                img.paste(bg, (x, y))

                font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf', h)
                _, _, textWidth, textHeight = font.getbbox(self.textLines[i])
                if textWidth < w:
                    textWidth = w
                textBox = Image.new(mode="RGBA", size=(textWidth, textHeight ), color=(0, 0, 0, 0))
                d = ImageDraw.Draw(textBox)
                d.text((0, 0), self.textLines[i], font=font, fill=(0, 0, 0))
                textBox.thumbnail((w, 1000  ), Image.ANTIALIAS)
                textBox = textBox.crop((0, 0, w, h))

                img.paste(textBox, (x, y), textBox.convert("RGBA"))


    def translate(self) -> None:
        text = ''
        for line in self.lines:
            text += line.getText() +  ' __1 ' 

        tText = (googletrans.Translator()
                    .translate(text, dest='vi').text)

        tText = re.sub('\\s+', ' ', tText).strip()
        self.textLines = re.split('__', tText)

        for i in range(len(self.textLines)):
            if len(self.textLines[i]) > 1 and self.textLines[i][0] == '1' :
                self.textLines[i] = self.textLines[i][1:]

    def getText(self) -> Str:
        text = ''
        for line in self.lines:
            text += line.text +  ' __1 ' 
        return text


# ====================================== Main ==============================================
img = None
paragraphs = None

def translate(imagePath, savePath):
    print("==========================",imagePath,"============================")
    tt = time.time()
    st = time.time()
    global paragraphs

    (contours, cContours, bImg) = getRawData(imagePath)
    print("getRawData: ", time.time() - st)
    
    st = time.time()
    boxes = getBoxes(contours, cContours, bImg)
    print("getBoxes: ", time.time() - st)
    for box in boxes:
        print(box.stPoint)
        # if box.confident>50:
        cv2.rectangle(img, box.stPoint, box.enPoint, (0, 255, 0), 1)
    cv2.imwrite(savePath, img)

    return 0
    st = time.time()    
    boxesLv1 = getBoxesLv1(boxes)

    print("getBoxesLv1: ", time.time() - st)
    boxesLv2 = getBoxesLv2(boxesLv1)
    
    st = time.time()    
    boxesLv2ToJpg(boxesLv2)
    print("saveAreasToJpg: ", time.time() - st)
    


    st = time.time()
    words = recognized(boxesLv2)
    print("recognized: ", time.time() - st)

    st = time.time()
    missBoxes = getMissBoxes(words)
    # rmInvalidWords(words)
    print("getMissBoxes: ", time.time() - st)
    
    st = time.time()
    missBoxesLv1 = getBoxesLv1(missBoxes)
    missBoxesLv2 = getBoxesLv2(missBoxesLv1)
    print("getAreas: ", time.time() - st)
    
    st = time.time()
    boxesLv2ToJpg(missBoxesLv2)
    print("saveAreasToJpg: ", time.time() - st)

    st = time.time()
    words += recognized(missBoxesLv2)
    print("recognized: ", time.time() - st)

    for boxLv1 in words:
        # print(boxLv1.w, boxLv1.h)
        cv2.putText(img, boxLv1.text, boxLv1.stPoint, 1, 1, (0, 255, 0))
        cv2.rectangle(img, boxLv1.stPoint, boxLv1.enPoint, (0, 255, 0), 1)
    cv2.imwrite(savePath, img)
    print(len(boxesLv1))
    return 0

    validWords  = getValidWords(words)
    




  
    st = time.time()
    lines = getLines(words)
    print("getLines: ", time.time() - st)

  
    for line in lines:
            # cv2.putText(img, line.text, line.stPoint, 1, 1, (255, 0, 255))
            cv2.rectangle(img, line.stPoint , line.enPoint, (0, 255, 0), 1)
    #     t += 1
    cv2.imwrite(savePath, img)
    return 0

# # ======================================================================================
    st = time.time()
    paragraphs = getParagraphs(lines)
    print("getParagraphs: ", time.time() - st)
    
    # for paragraph in paragraphs:
    #     for line in paragraph.lines:
    #         cv2.putText(img, str(t), line.stPoint, 1, 1, (255, 0, 255))
    #         cv2.rectangle(img, line.stPoint , line.enPoint, (0, 255, 0), 1)
    #     t += 1
    # cv2.imwrite(savePath, img)
    # return 0
    st = time.time()
    outputImg = Image.open(imagePath).convert("RGB")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futureDraw = {executor.submit(transText,  id, paragraphs[id].getText()): id for id in range(len(paragraphs))}
        for future in concurrent.futures.as_completed(futureDraw):
            id, text = future.result()
            paragraphs[id].textLines = text

    for paragraph in paragraphs:
        paragraph.draw(outputImg)
    outputImg.save(savePath)
    print("Translate & draw: ", time.time() - st)


    print("                       Total time: ", time.time() - tt)

    print("===========================================================================")


def getRawData(imagePath) -> Tuple:
    global img

    img = cv2.imread(imagePath)
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bImg = cv2.threshold(gImg, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)[1]
    cBImage = np.where(bImg == 0 , 255, 0).astype('ubyte')
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    dilation = cv2.dilate(bImg, rect_kernel, iterations = 1)
    cDilation = cv2.dilate(cBImage, rect_kernel, iterations = 1)
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    cContours = cv2.findContours(cDilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    return (contours, cContours, bImg)


def getBoxes(contours, cContours, bImg) -> List: 
    boxes = []
    rects = getBoundingRect(contours)

    # for rect in rects:
    #     x, y, w, h = rect
    #     white = np.count_nonzero(bImg[y:y+h,x:x+w] == 0)
    #     if (float(white) / float(w * h) >= 0.1
    #         and (w < bImg.shape[1]
    #             and h < bImg.shape[0])
    #     ):
    #         boxes.append(Box(rect, 0))

    cRects = getBoundingRect(cContours)
    for cRect in cRects:
        x, y, w, h = cRect
        # white = np.count_nonzero(bImg[y:y+h, x:x+w] == 0)
        # if (float(white) / float((w * h)) >= 0.1
        #     and (w < bImg.shape[1]
        #         and h < bImg.shape[0])
        # ):
        boxes.append(Box(cRect, 1))
    boxes = sorted(boxes, key= lambda x: (x.bgColor, x.stPoint[0]))
    return boxes



def getBoundingRect(contours) -> List:    
    result = []
    for cnt in contours:
        result.append(cv2.boundingRect(cnt))
    return result


def getBoxesLv1(boxes) -> List:
    boxesLv1 = []

    for box in boxes:
        inserted = False
        for boxLv1 in boxesLv1:
            if boxLv1.insertBox(box):
                inserted = True
                break
        if not inserted:
            boxesLv1.append(BoxLv1(box))
    result = []
    for boxLv1 in boxesLv1:
        if ((boxLv1.h/img.shape[0] > 0.01) 
            and boxLv1.w/img.shape[1] > 0.02):
            result.append(boxLv1)
    result = sorted(result, key=lambda x: (x.stPoint[1]))
    return result


def getBoxesLv2(boxesLv1):
    boxesLv2 = []
    for boxLv1 in boxesLv1:
        inserted = False
        for boxLv2 in boxesLv2:
            if boxLv2.insertBox(boxLv1):
                inserted = True
                break
        if not inserted:
            boxesLv2.append(BoxLv2(boxLv1))
    return boxesLv2

def getContrastColor(rbg) -> List:
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img) -> List:
    contrastImg = img.copy()
    for y in range(len(contrastImg)):
        for x in range(len(contrastImg[y])):
            contrastImg[y][x] = getContrastColor(contrastImg[y][x])
    return contrastImg    


def boxesLv2ToJpg(boxesLv2) -> None:
    for boxLv2 in boxesLv2:
        boxLv2.saveToJpg()

def recognized(boxesLv2) -> List:
    words = []
    for boxLv2 in boxesLv2:
        x, y, w, h = boxLv2.getRect()
        im = cv2.imread(boxLv2.imagePath)
        data = pytesseract.image_to_data(im, output_type=Output.DICT)
        words += toWords(data, x, y, boxLv2.bgColor)
    return words

def toWords(imgData,  sX, sY, bgColor) -> List:
    words = []
    for i in range(len(imgData['text'])):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])
        if(imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
            and w < img.shape[0]
            and h < img.shape[1]
            and imgData['conf'][i] > -1
        ): 
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', 
                            imgData['text'][i])
            words.append(Word([sX + x, sY + y, w, h], 
                                bgColor,
                                imgData['text'][i],
                                imgData['conf'][i]
                                ))
    return words

def getMissBoxes(words) -> List:
    missBoxes = []
    for word in words:
        if word.confident < 50:
            missBoxes.append(Box(word.getRect(), (word.bgColor + 1) % 2))
    return missBoxes 

def rmInvalidWords(words) -> None:
    for word in words:
        if word.confident < 50:
            words.remove(word)

def getValidWords(words) -> List:
    words = sorted(words, key=lambda x: (x.cePoint[1]))
    validWords = words[:]

    for vWord in validWords:
        for word in words:
            if  vWord != word:
                if(
                    vWord.stPoint[0] <= word.stPoint[0]
                    and vWord.stPoint[1] <= word.stPoint[1]
                    and vWord.enPoint[0] >= word.enPoint[0]
                    and vWord.enPoint[1] >= word.enPoint[1]
                ):
                    validWords.remove(vWord)
                    words.remove(vWord)
                    break
                # if( ((abs(vWord.cePoint[1] - word.cePoint[1]) <  vWord.h/2
                #         or abs(vWord.cePoint[1] - word.cePoint[1]) <  word.h/2)
                #     and (
                #         abs(vWord.cePoint[0] - word.cePoint[0]) < vWord.w/2
                #         or abs(vWord.cePoint[0] - word.cePoint[0]) < word.w/2
                #     )
                # )):
                #     if vWord.confident > word.confident:
                #         validWords.remove(word)
                #         words.remove(word)
                #     else:
                #         validWords.remove(vWord)
                #         words.remove(vWord)
                #     break
    validWords= sorted(validWords, key=lambda x: (x.stPoint[0]))
    return validWords

def getLines(words) -> list:
    lines = []
    for word in words:
        if word.confident >= 0:
            inserted = False
            for line in lines:
                if line.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                lines.append(Line(word))
    lines = sorted(lines, key= lambda key: key.cePoint[1])
    return lines 

# def updateLineRect(self, lines):


def getParagraphs(lines) -> List:
    paragraphs = []

    for line in lines:
            inserted = False
            for paragraph in paragraphs:
                if paragraph.insertLine(line):
                    inserted = True
                    break 
            if not inserted:
                paragraphs.append(Paragraph(line))

    return paragraphs


def transText(id, text) -> None:
    tText = (googletrans.Translator()
                .translate(text, dest='vi').text)

    tText = re.sub('\\s+', ' ', tText).strip()
    textLines = re.split('__', tText)

    for i in range(len(textLines)):
        if len(textLines[i]) > 1 and textLines[i][0] == '1' :
            textLines[i] = textLines[i][1:]
    return id, textLines



def draw(paragraph, outputImg):
    paragraph.draw(outputImg, 55)

translate('images/1.png', 'output/1.png')
translate('images/2.png', 'output/2.png')
translate('images/3.png', 'output/3.png')
translate('images/4.png', 'output/4.png')
translate('images/5.png', 'output/5.png')
translate('images/6.png', 'output/6.png')
translate('images/7.png', 'output/7.png')
translate('images/8.png', 'output/8.png')
translate('images/9.png', 'output/9.png')
translate('images/10.png', 'output/10.png')
translate('images/11.png', 'output/11.png')
translate('images/12.png', 'output/12.png')
translate('images/13.png', 'output/13.png')
translate('images/14.png', 'output/14.png')


