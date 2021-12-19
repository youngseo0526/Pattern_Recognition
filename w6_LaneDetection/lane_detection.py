import cv2
import pipeline

img = cv2.imread('./test_images/solidWhiteRight.jpg')

result = pipeline.run(img)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.imwrite('result__.png', result)
cv2.destroyAllWindows()