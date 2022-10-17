# OpenCV

## imread - imshow - imwrite

Reading and writing image:

```python
img = cv2.imread(path, flags)
cv2.imwrite(path, img, params)
```

Flag determine if image should be loaded as grey, with alpha channel, ect ([IMREAD IMWRITE flags](https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html)).  
Function `cv2.imread` returns `np.ndarray` with shape $(height \times width \times channels)$ (if loaded grey, channel dimention is skiped). Because of this, you can change image like you change numpy array:

```python
img[100:300, 650:950] = img2[400:500, 700:1000]
```

### Showing images

If you plot image with `plt.imshow(img)`, it could change color scale and the result will be different than real.  
To control image windows closure use above:

```python
cv2.waitKey(0)  # wait until any key has been pushed
cv2.waitKey(5000)  # wait 5sec or until any key has been pushed
cv2.destroyAllWindows()  # close all windows
```

## Resizing

You can change size of image providing shape pixels or xy fractions:

```python
img = cv2.resize(img, (400, 300))  # based on pixels
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # based on xy fractions
```

## Rotating and flipping

```python
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.flip(img, flag)
```

`flip` flag values:

- 0 - flip around x-axis
- 1 - flip around y-axi
- -1 - flip around both axis

## Video capture

To enable capture view from your computer camera use:

```python
cap = cv2.VideoCapture(0)
```

To read from it and show it as a video you read it as image and show it in loop:

```python
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
```

At the end of loop declare if statement escaping loop after clicking *q*. Argument of `cv2.waitKey` determines how many milliseconds to wait to read for pressing any key, if no key was pressed returns -1, otherwise returns code of key.  

To get video property by [property id](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d) use:

```python
prop = vid.get(prop_id)
```

## Painting and texting

Common usecases showed in [tutorial4](tutorial4.py).  

## Color space changing

To convert image color space use [convertion code](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0) as below:

```python
new_img = cv2.cvtColor(img, convertion_code)
```

## Thresholding and masking

To get mask based on pixel value thresholding, define boundries and use `inRange` function:

```python
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
```

To apply mask use:

```python
new_img = cv2.bitwise_and(img, img, mask=mask)
```

Remember that result of thresholding will differ depending on color scale type (rba, hsv, ect.)

## Finding important features

OpenCV has also tools to finding corners. In computer vision corners are treated as more intresting than lines or spaces due to defining the shapes of objects and objects like face having them more than flat surfaces. Because of it we treat them as features. To find them we can use few technics, for example:

- [Harris corner detection](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html) - obraz traktujemy filtrami sobela: pionowym i poziomym, następnie dla małych jego wycinków zaznaczamy gradienty na osi xy, w ten sposób uzyskujemy wykres punktowy gradientów fragmentu; zakładamy, że jeżeli gradienty są bliskie zeru, wtedy mamy do czynienia z płaską przestrzenią, jeżeli gradienty układają się w linię, wtedy mamy do czynienia z krawędzią, a jeżeli są rozproszone, wtedy jest to narożnik; aby wyrazić to przy pomocy wzoru matematycznego, zakładając, że każdy punkt "waży" tyle samo, tworzymy elipsę opisującą rozkład punktów; następnie wyrażamy ją przy pomocy dwóch prostopadłych wektorów (wektorów własnych) promień $\lambda_{1}, \lambda_{2}$; na podstawie ich wielkości określamy:
- $\lambda_{1} \lambda_{2} << threshold \rightarrow$ *flat region*,
- $\lambda_{1} >> \lambda_{2} \vee \lambda_{1} << \lambda_{2} \rightarrow$ *line*,
- $\lambda_{1} \approx \lambda_{2} \rightarrow$ *corner*.

[Youtube explanation](https://www.youtube.com/watch?v=Z_HwkG90Yvw&ab_channel=FirstPrinciplesofComputerVision). Code:

```python
cv2.cornerHarris(img, blockSize, ksize, k)
```

- [Shi-Tomasi corner detection](https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html) - metoda prawie identyczna jak poprzednia, z tym, że ostatecznie pod uwagę brany jest krótszy wektor własny; metoda ta działa lepiej dlatego jest przeważnie stosowana; kod:

```python
cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
```

## Template matching

Template matching is the process of finding same looking part of image as the template (image we want to find). We doing it similar to subtracting template values from moving window and squaring it. Lower score imply good match. Instead of calculating entire expression, we only take part of it we need to be maximized and it's same as convolution but with no flipping. This part of expression is called *cross-correlation*. To make it more accurate (template and photo could have different lighting) we normalize it dividing by multiplication of root quared values of template and part of image. [Link to youtube explanation](https://www.youtube.com/watch?v=1_hwFc8PXVE&ab_channel=FirstPrinciplesofComputerVision). Code:

```python
result = cv2.matchTemplate(img2, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
```

Match template method is formula which is minimized/maximized ([link](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d)).

## Cascade Classifiers

You can use *CascadeClassifier* with predefined params from *.xml* file to recognize objects like face. Code example:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
```
