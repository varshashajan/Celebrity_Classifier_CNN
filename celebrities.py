import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

image_dir = r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped'
lionel_messi = os.listdir(r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped\lionel_messi')
maria_sharapova = os.listdir(r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped\maria_sharapova')
roger_federer = os.listdir(r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped\roger_federer')
serena_williams = os.listdir(r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped\serena_williams')
virat_kohli = os.listdir(r'D:\3rd sem\deeplearning\practical assignment\Dataset_Celebrities\cropped\virat_kohli')
print("---------------------------------------------------")


dataset = []
label = []
img_size = (128,128)

print('The length of lionel messi image is',len(lionel_messi))
print('The length of maria_sharapova  image is',len(maria_sharapova))
print('The length of roger_federer image is',len(roger_federer))
print('The length of serena_williams image is',len(serena_williams))
print('The length of virat_kohli image is',len(virat_kohli))


dataset = []
label = []
img_size = (128,128)

for i , image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger_federer),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)

dataset=np.array(dataset)
label = np.array(label)

print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))

print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(128,128,3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),  # Additional dense layer
  tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  
            metrics=['accuracy'])

print("--------------------------------------\n")
print("Training Started.\n")

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)

history=model.fit(x_train,y_train,epochs=20,validation_split=0.1,callbacks=[early_stop])
print("Training Finished.\n")
print("--------------------------------------\n")

print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction,axis=1)[0]
    class_name = ['lionel_messi','maria_sharapova','roger_federer','serena_williams','virat_kohli']
    predicted_class_name = class_name[predicted_class]
    return predicted_class_name

print(make_prediction('Dataset_Celebrities\cropped\maria_sharapova\maria_sharapova11.png',model))
print('--------------------------------------------------------')