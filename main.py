import string
from typing import List
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


class Box:
    def __init__(self, rect, bgColor) -> None:
        self.startPoint = [int(rect[0]), int(rect[1])]
        self.endPoint = [int(rect[0] + rect[2]), int(rect[1] + rect[3])]
        self.centerPoint = [rect[0] + rect[2]/2, rect[1] + rect[3] / 2]
        self.width = rect[2]
        self.height = rect[3]
        self.bgColor = bgColor


class BigBox:
    def __init__(self, box:Box) -> None:
        self.X_NOUN = 0.6
        self.Y_NOUN = 0.3
        self.startPoint = box.startPoint[:]
        self.endPoint = box.endPoint[:]
        self.centerPoint = box.centerPoint[:]
        self.width = box.width
        self.height = box.height
        self.bgColor = box.bgColor
        self.epY = self.height * self.Y_NOUN
        self.epX = self.height * self.X_NOUN
    
    def __isValid(self, box:Box) -> Boolean:
        if(box.bgColor == self.bgColor
            and abs(self.endPoint[0] - box.startPoint[0]) <= self.epX
            and abs(self.height - box.height) <= self.epY * 2
            and (abs(self.startPoint[1] - box.startPoint[1]) <= self.epY
                    or abs(self.centerPoint[1] - box.centerPoint[1]) <= self.epY
                    or abs(self.endPoint[1] - box.endPoint[1]) <= self.epY 
                )
        ):
            return True
        return False
    
    def __update(self, box:Box) -> None:
        if self.startPoint[0] > box.startPoint[0]:
            self.startPoint[0] = box.startPoint[0]
        if self.startPoint[1] > box.startPoint[1]:
            self.startPoint[1] = int(box.startPoint[1])

        if self.endPoint[1] < box.endPoint[1]:
            self.endPoint[1] = box.endPoint[1]
        if self.endPoint[0] < box.endPoint[0]:
            self.endPoint[0] = box.endPoint[0]
        
        self.centerPoint[0] = (self.startPoint[0] + self.endPoint[0])/2
        self.centerPoint[1] =  (self.startPoint[1] + self.endPoint[1])/2

        self.height = self.endPoint[1] - self.startPoint[1]
        self.width = self.endPoint[0] - self.startPoint[0]

        self.epX = self.height *  self.X_NOUN
        self.epY = self.height * self.Y_NOUN
        
   

    def mergeBox(self, box:Box) -> Boolean:
        if self.__isValid(box):
            self.__update(box)
            return True
        return False

    def getRect(self, imgShape):
        x = self.startPoint[0]
        y = int(self.startPoint[1] - (self.epY/2 
                                    if(self.startPoint[1] - self.epY/2 >= 0) 
                                    else 0))
        endX = self.endPoint[0]
        endY = int(self.endPoint[1] + (self.epY/2 
                                    if(self.endPoint[1] - self.epY/2 <= imgShape[1]) 
                                    else 0))

        return [x, y, endX, endY]




class Word:
    def __init__(self, box, confi, text, bgColor) -> None:
        self.text = text
        self.startPoint = [box[0], box[1]]
        self.endPoint = [box[0] + box[2], box[1] + box[3]]
        self.centerPoint = [box[0] + box[2] * 0.5, box[1] + box[3] * 0.5]
        self.width = box[2]
        self.height = box[3]
        self.confident = int(confi)
        self.bgColor = bgColor



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
        if text.isupper():
            text = text.lower()
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






# def filterTheboxes

def getBoundingRect(contours) -> List:    
    result = []
    for cnt in contours:
        result.append(cv2.boundingRect(cnt))
    return result

def getBoxes(contours, cContours, binaryImg) -> List:
    boxes = []
    rects = getBoundingRect(contours)

    for rect in rects:
        x, y, w, h = rect
        white = np.count_nonzero(binaryImg[ y:y+h,x:x+w] == 0)
        if (float(white) / float(w * h) >= 0.1
            and (w < binaryImg.shape[0]
                and h < binaryImg.shape[1])
        ):
            boxes.append(Box(rect, 0))

    cRects = getBoundingRect(cContours)
    for cRect in cRects:
        x, y, w, h = cRect
        white = np.count_nonzero(binaryImg[y:y+h, x:x+w] == 1)
        if (float(white) / float((w * h)) >= 0.1
            and (w < binaryImg.shape[0]
                and h < binaryImg.shape[1])
        ):
            boxes.append(Box(cRect, 1))

    return boxes

def isChildBox(pBox, cBox):
    if(pBox.startPoint[0] <= cBox.startPoint[0]
        and pBox.startPoint[1] <= cBox.startPoint[1]
        and pBox.endPoint[0] >= cBox.endPoint[0]
        and pBox.endPoint[1] >= cBox.endPoint[1]
        ):
        return True
    return False

def removeChildBoxes(boxes):
    # result = boxesl
    for pBox in boxes:
        for cBox in boxes:
            if cBox != pBox and isChildBox(pBox, cBox):
                boxes.remove(cBox)

def getBigBoxes(boxes) -> List:
    bigBoxes = []

    for box in boxes:
        isMerged = False
        for bigBox in bigBoxes:
            if bigBox.mergeBox(box):
                isMerged = True
                break
        if not isMerged:
            bigBoxes.append(BigBox(box))
    return bigBoxes



def getImgData(bigBox, img):
    sX, sY = bigBox.startPoint
    eX, eY = bigBox.endPoint
    print("get image data")
    pytesseract.image_to_data(img[sY:eY, sX:eX])






def toWords(imgData, imgSize, sX, sY, bgColor) -> List:
    words = []
    for i in range(len(imgData['text'])):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])

        if(imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
            and w < imgSize[0]
            and h < imgSize[1]
            and imgData['conf'][i] > -1
        ): 
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', 
                            imgData['text'][i])
            words.append(Word([sX + x, sY + y, w, h], 
                                imgData['conf'][i], 
                                imgData['text'][i], 
                                bgColor))
    return words



def getContrastColor(rbg):
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img):
    contrastImg = img.copy()
    for y in range(len(contrastImg)):
        for x in range(len(contrastImg[y])):
            contrastImg[y][x] = getContrastColor(contrastImg[y][x])
    return contrastImg    












def main(imgPath, savePath):
    st = time.time()
    img = cv2.imread(imgPath)
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)[1]
    contrastBImage = np.where(binaryImg == 0 , 255, 0).astype('ubyte')
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    dilation = cv2.dilate(binaryImg, rect_kernel, iterations = 1)
    cDilation = cv2.dilate(contrastBImage, rect_kernel, iterations = 1)
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    cContours = cv2.findContours(cDilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]

    boxes = getBoxes(contours, cContours, binaryImg)
    boxes = sorted(boxes, key= lambda x: (x.bgColor, x.startPoint[0]))
    bigBoxes = getBigBoxes(boxes)
    bigBoxes = sorted(bigBoxes, key=lambda x: (x.startPoint[1]))

    # for box in boxes:
    #     cv2.rectangle(img, box.startPoint, box.endPoint, (0, 255, 0), 1)
    # cv2.imwrite(savePath, img)

    # return 0
    words = []
    t = 0
    for bigBox in bigBoxes:
        x, y, w, h = bigBox.getRect(img.shape)
        im = None
        if bigBox.bgColor == 0:
            im = img[y:h, x:w]
        else:
            im = getContrastImg(img[y:h, x:w])
        cv2.imwrite('cache/' + str(t)+'.jpg', im)
        t += 1

    for i in range(t):
        x, y, w, h = bigBoxes[i].getRect(img.shape)
        im = cv2.imread('cache/'+str(i)+'.jpg')
        data = pytesseract.image_to_data(im, output_type=Output.DICT)
        words += toWords(data, img.shape, x, y, 1)


    nBoxes = []
    for word in words:
        x, y = word.startPoint
        w, h = word.endPoint
        if word.confident < 50 :
            nBoxes.append(Box((x, y, w-x, h-y), (word.bgColor + 1)%2))

    nBigBoxes = getBigBoxes(nBoxes)
    t = 0
    for nBigBox in nBigBoxes:
        x, y, w, h = nBigBox.getRect(img.shape)
        im = None
        if nBigBox.bgColor == 0:
            im = img[y:h, x:w]
        else:
            im = getContrastImg(img[y:h, x:w])
        cv2.imwrite('cache/second_' + str(t)+'.jpg', im)
        t += 1

    for i in range(t):
        x, y, w, h = nBigBoxes[i].getRect(img.shape)
        im = cv2.imread('cache/second_'+str(i)+'.jpg')
        data = pytesseract.image_to_data(im, output_type=Output.DICT)
        words += toWords(data, img.shape, x, y, 1)


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


    # cv2.imwrite(savePath, img)
    print("Excuse time: ", time.time() - st)




main('images/1.png', 'output/aa1.png')
# main('images/2.png', 'output/aa2.png')
# main('images/3.png', 'output/aa3.png')
# main('images/4.png', 'output/aa4.png')
# main('images/5.png', 'output/aa5.png')
# main('images/6.png', 'output/aa6.png')
# main('images/7.png', 'output/aa7.png')
# main('images/8.png', 'output/aa8.png')
# main('images/9.png', 'output/aa9.png')
# main('images/11.png', 'output/aa11.png')
# main('images/12.png', 'output/aa12.png')
# main('images/13.png', 'output/aa13.png')

# main('images/14.png', 'output/aa14.png')
# main('images/15.png', 'output/aa15.png')