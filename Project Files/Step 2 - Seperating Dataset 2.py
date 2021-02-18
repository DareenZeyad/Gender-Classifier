# Step 2 - Separating Dataset 2
import os
import shutil

data = []

cntrmale = 0
cntrfemale = 0

fileList = os.listdir('part1')
for filename in fileList:
    if filename.endswith('.jpg'):
        filenameList = filename.split('_')
        if filenameList[1] == '0':
            if cntrmale % 10 == 8 or cntrmale % 10 == 9:
                shutil.move('part1/' + filename, 'part1/test_set/male/' + filename)
            else:
                shutil.move('part1/' + filename, 'part1/training_set/male/' + filename)
            cntrmale += 1
        else:
            if cntrfemale % 10 == 8 or cntrfemale % 10 == 9:
                shutil.move('part1/' + filename, 'part1/test_set/female/' + filename)
            else:
                shutil.move('part1/' + filename, 'part1/training_set/female/' + filename)
            cntrfemale += 1