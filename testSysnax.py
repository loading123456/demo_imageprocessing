from xmlrpc.client import Boolean


class Word:
    def __init__(self, box, text) -> None:
        self.text = text
        self.startPoint = [box[0], box[1]]
        self.endPoint = [box[0] + box[2], box[1] + box[3]]
        self.center = [(box[0] + box[2]) / 2, (box[1] + box[3]) * 0.5]



class Line:
    def __init__(self, word:Word) -> None:
        self.words = [word]
        self.wordsSize = 1
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.center = word.center
        self.epsilon = (self.endPoint[1] - self.startPoint[1]) * 0.2
        self.spaceSize = (self.endPoint[1] - self.startPoint[1]) * 0.3

    def __updateLine(self, position: int) -> None:
        if self.startPoint[0] > self.words[position].startPoint[0]:
            self.startPoint[0] = self.words[position].startPoint[0]
        if self.startPoint[1] > self.words[position].startPoint[1]:
            self.startPoint[1] = self.words[position].startPoint[1]
        if self.endPoint[0] < self.words[position].endPoint[0]:
            self.endPoint[0] = self.words[position].endPoint[0]
        if self.endPoint[1] < self.words[position].endPoint[1]:
            self.endPoint[1] = self.words[position].endPoint[1]
        
        self.center[0] = (self.center[0] + self.words[position].center[0]) / 2
        self.center[1] = (self.center[1] + self.words[position].center[1]) / 2

        self.epsilon = (self.endPoint[1] - self.startPoint[1]) * 0.2
        self.spaceSize = (self.endPoint[1] - self.startPoint[1]) * 0.3
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
                if distance <= self.spaceSize * 2:
                    return 0
                return -1

            if position == self.wordsSize:
                distance = word.startPoint[0] - self.words[-1].endPoint[0]
                if distance <= self.spaceSize * 2:
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
            self.__updateLine(position)
            return True
        return False

    def showText(self):
        text = ''
        for i in range(len(self.words)):
            text += self.words[i].text + ' '
        print('size: ', self.startPoint, self.endPoint ,len(self.words), ' -- ',text)


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
        if line.startPoint[1] - self.lines[-1].endPoint[1] <= self.epsilon:
            if abs(line.startPoint[0] - self.lines[-1].startPoint[0]) <= self.epsilon * 2:
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

        

word0 = Word([0,0,50, 10], 'hello')
word2 = Word([0,12,15, 10], 'hi')
word4 = Word([18, 12, 20, 10], 'you')
word1 = Word([53,0,50, 10], 'what')
word3 = Word([110,0,90, 10], 'welcome')

lines = Lines()

lines.insertWord(word0)
lines.insertWord(word2)
lines.insertWord(word4)
lines.insertWord(word1)
lines.insertWord(word3)


lines.sort()
# print(lines.lines[0].startPoint, lines.lines[0].endPoint)
# lines.show()

textBoxs = TexBoxs()

for line in lines.lines:
    textBoxs.insertLine(line)

textBoxs.show()