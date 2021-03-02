from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import cv2 as cv
from keras.models import load_model
import numpy as np
import json
import base64
from PIL import Image
import io
# import cStringIO
import re

import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


class Digits(APIView):
    """
    List all snippets, or create a new snippet.
    """
    def get(self, request, format=None):
        pass
        # snippets = Snippet.objects.all()
        # serializer = SnippetSerializer(snippets, many=True)
        # return Response(serializer.data)
        return Response("cyka")

    def post(self, request, format=None):
        data = json.loads(request.body.decode('utf-8'))
        datauri = data['image']
        imgstr = re.search(r'base64,(.*)', datauri).group(1)
        image_bytes = io.BytesIO(base64.b64decode(imgstr))
        im = Image.open(image_bytes)
        image = np.array(im)
        image = (image.astype(np.uint8))
        image = np.delete(image, (2), axis=2)
        image = np.float32(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv.imwrite("qwerty1.png", image)
        _, thresh = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
        # cv.imwrite("qwerty.png", thresh)
        thresh = thresh.astype(np.uint8)
        contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        # Iterate through all the 
        # print(len(contours))
        array, digits = [], []

        for contour in contours:
            x,y,w,h = cv.boundingRect(contour)
            array.append({'x':x, 'y':y, 'w':w, 'h':h})

        modelPath = BASE_DIR+'/final_model.h5'

        model = load_model(modelPath)
        # print(hierarchy, hierarchy[0][1][3])
        for j,i  in enumerate(array) :
            if hierarchy[0][j][3] == -1:
                tempImg = thresh[i['y']:i['y']+i['h'], i['x']:i['x']+i['w']]
                # making images square
                (a,b)=tempImg.shape
                if a>b:
                    padding=((0,0),((a-b)//2,(a-b)//2))
                else:
                    padding=(((b-a)//2,(b-a)//2),(0,0))
                tempImg = np.pad(tempImg,padding,mode='constant',constant_values=0)

                resized = cv.resize(tempImg, (22,22), interpolation = cv.INTER_AREA)
                resized = np.pad(resized,3,mode='constant',constant_values=0)
                # resized = cv.resize(tempImg, (28,28), interpolation = cv.INTER_NEAREST)
                # print(resized,"zzzzzzzzzzzzzzzzz")
                # resized = np.pad(resized, 4, )
                # kernel = np.ones((2,2),np.uint8)
                # resized = cv.morphologyEx(resized, cv.MORPH_OPEN, kernel)
                # resized = cv.morphologyEx(resized, cv.MORPH_CLOSE, kernel)
                # cv.imwrite("qwerty"+str(i)+".png", resized)
                resized = resized.reshape((1,28,28,1))
                digits.append(model.predict_classes(resized))

        # for i in digits:
        #     print(i)
        # cv.imshow('img',imgMat)
        return Response(digits)