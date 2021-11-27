import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import cv2
import os
from scipy import ndimage,misc
import scipy as scipy
import imageio
brmatr = []

def getbrmatr(image):
    arr = np.array(image)  # Parsing pixels as array
    height, width = image.size
    for i in range(width):
        for j in range(height):
            brmatr.append(np.sqrt((0.299 * arr[i][j][0] ** 2) + (0.587 * arr[i][j][1] ** 2) + (0.114 * arr[i][j][2] ** 2)))  # Converting RGB to Brightness

def QuickSort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)

        return QuickSort(less)+equal+QuickSort(greater)
    else:
        return array

def ftAlphaTrimmedMean(image):
    # deep copy
    img = image.copy()

    # Get image height and width
    height, width = image.size

    height = height - 1
    width = width - 1
    imarr = np.array(img)
    a = 0

    # loop through
    for i in range(1, width):
        for j in range(1, height):
            arr = []

            # get pixel value and append it to array
            a = imarr[i - 1][j - 1]
            arr.append(a)

            a = imarr[i - 1][j]
            arr.append(a)

            a = imarr[i - 1][j + 1]
            arr.append(a)

            a = imarr[i][j - 1]
            arr.append(a)

            a = imarr[i][j]
            arr.append(a)

            a = imarr[i][j + 1]
            arr.append(a)

            a = imarr[i + 1][j - 1]
            arr.append(a)

            a = imarr[i + 1][j]
            arr.append(a)

            a = imarr[i + 1][j + 1]
            arr.append(a)

            # Sorting
            arr = QuickSort(arr)
            leng = len(arr) - 1

            # get minddle index
            middleIndex = int(leng / 2)

            total = 0

            total += arr[middleIndex - 2]
            total += arr[middleIndex - 1]
            total += arr[middleIndex]
            total += arr[middleIndex + 1]
            total += arr[middleIndex + 2]

            total = int(total / 5)

            # set pixel value back to image
            imarr[i][j] = total
    img = Image.fromarray(imarr)
    return img

def print_image(path):
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()

def hist(image):
    his = [0 for i in range(256)]
    brightnessarray = np.asarray(brmatr, dtype=int)
    for i in range(len(brightnessarray)):
        a = brightnessarray[i]
        his[a]+=1
    W = len(his)  # кол-во элементов массива
    hist = Image.new("RGB", (W, 100), "white")  # создаем рисунок в памяти
    draw = ImageDraw.Draw(hist)  # объект для рисования на рисунке
    maxx = float(max(his))  # высота самого высокого столбика
    if maxx == 0:  # столбики равны 0
        draw.rectangle(((0, 0), (W, 100)), fill="black")
    else:
        for i in range(W):
            draw.line(((i, 100), (i, 100 - his[i] / maxx * 100)), fill="black")  # рисуем столбики
    del draw  # удаляем объект
    hist.save('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\Results\\Hist.png')

def grayscale(image):
    height, width = image.size
    arr = np.asarray(brmatr, dtype=np.uint8)  # Converting list to numpy array
    reshaped_brmatr = arr.reshape(width,height)  # reshaping it

    print(reshaped_brmatr)  # Printing brightness matrix

    grayscaleimg = Image.fromarray(reshaped_brmatr)  # Saving it as new image a.k.a. grayscale
    grayscaleimg.save("C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\Results\\grayscale.png")

    return grayscaleimg

def binarization(image):
    bw = []  # Converting to pure black and white
    height, width = image.size
    for i in range(height * width):
        if (brmatr[i] <= 128):
            bw.append(0)
        else:
            bw.append(255)

    bwarray = np.asarray(bw, dtype=np.uint8)  # Converting list to numpy array
    reshapedbw = bwarray.reshape(width,height)  # reshaping it
    bwimage = Image.fromarray(reshapedbw)  # Saving it as new image a.k.a. black and white
    bwimage.save("C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\Results\\black and white.png")

def denoise(image):
    height, width = image.size
    imgdenoise = ftAlphaTrimmedMean(grayscale(image))
    imgdenoise.save('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\Results\\denoised.png')

#not used
def sobel():
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ascent = misc.ascent()
    result = ndimage.sobel(ascent)
    im1=imageio.imread('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\test.png')
    ax1.imshow(ascent)
    ax2.imshow(result)
    plt.show()


def edge_detections():
    img0 = imageio.imread('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\cat.jpg')

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray, (3, 3), 0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()


def main():
    image = Image.open('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\cat_noise.jpg')
    getbrmatr(image)
    #task 1
    print_image('C:\\Users\\Мария\\Desktop\\GraphicsAddTask\\cat_noise.jpg')
    #task 2
    hist(image)
    #task 3
    grayscale(image)
    #task 4
    binarization(image)
    #task 5+5
    denoise(image)
    #sobel()
    edge_detections()
main()