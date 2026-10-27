# NeuroCanvas — Academic Notes

## Proposed title
AI-Based Interactive Doodle and Handwriting Recognition Using Neural Networks

## Neural networks used
### 1. Convolutional Neural Network (CNN)
Used for doodle image classification.

Architecture:
Input 28x28x1
→ Conv2D(32)
→ MaxPooling
→ Conv2D(64)
→ MaxPooling
→ Flatten
→ Dense(128)
→ Dropout
→ Softmax(6 classes)

### 2. Neural OCR
EasyOCR uses neural networks to detect and recognize text from the written image.

## Why CNN for doodles?
Doodles are images. CNNs learn spatial patterns such as edges, curves and shapes while sharing weights across the image.

## Why this is a real NN project
The doodle model is trained locally with a labeled dataset. The project can report training accuracy, validation accuracy, test accuracy, loss and a confusion matrix.

## Demo
1. Open Doodle Mode.
2. Draw a supported class.
3. Click Recognize Doodle.
4. Show prediction and class probabilities.
5. Switch to Write Mode.
6. Write a short sentence.
7. Click Convert Handwriting.
8. Edit recognized text if needed.
9. Export DOCX.

## Extension ideas
- Add more Quick, Draw! classes.
- Compare CNN with a simple ANN/MLP baseline.
- Add data augmentation.
- Plot accuracy/loss.
- Add a confusion matrix page.
- Save previous predictions.
