# Haar Cascade Classifiers

This directory should contain the Haar cascade XML files used for face and eye detection.

## Required Files

- `haarcascade_frontalface_default.xml` - For face detection
- `haarcascade_eye.xml` - For eye detection

## How to Get These Files

If these files are not present, you can download them from the OpenCV GitHub repository:

1. Face cascade: [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
2. Eye cascade: [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

Alternatively, you can run this command to download them automatically:

```bash
# From the project root directory
mkdir -p haarcascades
cd haarcascades
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
```

## Additional Information

These XML files are pre-trained Haar cascade classifiers for detecting objects in images, specifically faces and eyes in our application.

For more information on Haar cascades and how they work, refer to the [OpenCV documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).
