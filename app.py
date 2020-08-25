# starts here
# import necessary packages
import json
import uuid
from flask import Flask, request, Response

# import the necessary packages
from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2


# function to detect face
def face_detect(img):
    # load the digit classifier from disk
    print("[INFO] loading digit classifier...")
    model = load_model('digit_classifier.h5')

    # load the input image from disk and resize it
    print("[INFO] processing image...")
    # image = cv2.imread('sudoku_puzzle.jpg')
    image = imutils.resize(img, width=600)

    # find the puzzle in the image and then
    (puzzleImage, warped) = find_puzzle(image, debug=0)

    # initialize our 9x9 sudoku board
    board = np.zeros((9, 9), dtype="int")

    # a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []

        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=0)

            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # classify the digit and update the sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred

        # add the row to our cell locations
        cellLocs.append(row)

    # construct a sudoku puzzle from the board
    #print("[INFO] OCR'd sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    #puzzle.show()

    # solve the sudoku puzzle
    #print("[INFO] solving sudoku puzzle...")
    solution = puzzle.solve()
    #solution.show_full()

    # loop over the cell locations and board
    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box

            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY

            # draw the result digit on the sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # show the output image
    # cv2.imshow("Sudoku Result", puzzleImage)
    # cv2.waitKey(0)

    path_file = ('static/%s.jpg' % uuid.uuid4().hex)
    cv2.imwrite(path_file, puzzleImage)
    return json.dumps(path_file)

    '''
    face_cascade = cv2.CascadeClassifier('face_cascade.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255))

    # saving the file
    path_file = ('static/%s.jpg' % uuid.uuid4().hex)
    cv2.imwrite(path_file, img)
    return json.dumps(path_file)
'''


# API
app = Flask(__name__)


# route http post to this method
@app.route('/api/upload', methods=['POST'])
def upload():
    # retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # saving the file
    path_file = ('received/%s.jpg' % uuid.uuid4().hex)
    cv2.imwrite(path_file, img)

    # processing image
    img_processed = face_detect(img)

    # response
    print("[INFO] uploaded...")
    return Response(response=img_processed, status=200, mimetype="application/json")  # returning json string


# start server
app.run(host='0.0.0.0', port=5000, debug=True)
