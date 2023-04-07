import os # to dealing with Images folders in Train and Test 
import numpy as np # to dealing with arrays to convert Images To Array 
import pandas as pd # To dealing with csv files 
import matplotlib.pyplot as plt # To Plotting Loss and Accuracy and Show Images On Test Model 
from tensorflow.keras.models import Sequential # The Model We Will Use in our project 
from tensorflow.keras.callbacks import ModelCheckpoint # to save the model in the best case of accuracy 
from tensorflow.keras.layers import Conv2D,   , Dropout , Dense,Flatten # Model Layers 


path_train = "imgs/train"
classes = [c for c in os.listdir(path_train) if not c.startswith(".")]
classes.sort()
print(classes)


def create_data_as_csv(path, fname):
    class_names = os.listdir(path)
    data = []
    if(os.path.isdir(os.path.join(path,class_names[0]))):
        for cn in class_names:
            file_names = os.listdir(os.path.join(path,cn))
            for im in file_names:
                data.append({
                    "FileName":os.path.join(path,cn,im),
                    "ClassName":cn
                })
    else:
        cn = "test"
        file_names = os.listdir(path)
        for im in file_names:
                data.append({
                    "FileName":os.path.join(path,im),
                    "ClassName":cn
                })
    data = pd.DataFrame(data)
    data.to_csv(fname,index=False)
    
    
    
    
    
    create_data_as_csv("imgs/train", "train.csv")
create_data_as_csv("imgs/test", "test.csv")



train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



plt.figure(figsize=(15,5))
x = train_data["ClassName"].value_counts(sort=False).to_numpy()
y = train_data["ClassName"].value_counts(sort=False).index.to_numpy()
plt.bar(y,x,0.8888,color="green")
plt.xlabel("Classes")
plt.ylabel("Image Counts")
plt.show()




lables_list = list(set(train_data['ClassName'].values.tolist()))
lables_list.sort()
lables_indx = {ln:i for i,ln in enumerate(lables_list)}
print(lables_indx)



train_data["ClassName"].replace(lables_indx,inplace=True)




import pickle
with open(os.path.join("lables_list.pkl"),"wb") as handle:
    pickle.dump(lables_indx,handle)
    
    


from tensorflow.keras.utils import to_categorical
labels = to_categorical(train_data["ClassName"])
print(labels[:5])



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:,0],labels,test_size=0.2,random_state=42)




len(x_train),len(x_test),len(y_train),len(y_test)




from tensorflow.keras.preprocessing import image
def image_to_tensor(img_path):
    img = image.load_img(img_path,target_size=(64,64))
    x = image.img_to_array(img)
    return np.expand_dims(x,axis=0)
    
    
    
    
 def images_to_tensor(img_paths):
    list_of_tensors = [image_to_tensor(img_path) for img_path in (img_paths)]
    return np.vstack(list_of_tensors)
    
    
    
train_tensors = images_to_tensor(x_train).astype('float32')/255 - 0.5
test_tensors  = images_to_tensor(x_test).astype('float32') /255 - 0.5




model = Sequential()

model.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu',input_shape=(64,64,3),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(64,64,3),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128,kernel_size=2,padding='same',activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=256,kernel_size=2,padding='same',activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=512,kernel_size=2,padding='same',activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation='relu',kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax',kernel_initializer='glorot_normal'))

model.summary()
model.compile(optimizer = 'adam',loss="categorical_crossentropy",metrics=['accuracy'])





from tensorflow.keras.utils import plot_model
plot_model(model,to_file="plot.png",show_shapes=True,show_layer_names=True)




checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',period=1)
callbacks_list = [checkpoint]




model_history = model.fit(train_tensors,
                          y_train,
                          validation_data = (test_tensors, y_test),
                          epochs=18,
                          batch_size=40,
                          shuffle=True,
                          callbacks=callbacks_list
                          )
                          
                          
 
 
fig , (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.plot(model_history.history["loss"],color="b",label="Training Loss")
ax1.plot(model_history.history["val_loss"],color="r",label="Testing Loss")
ax1.set_xticks(np.arange(1,25,1))
ax1.set_yticks(np.arange(0,1,0.1))
ax1.title.set_text("LOSS")

ax2.plot(model_history.history["accuracy"],color="b",label="Training Accuracy")
ax2.plot(model_history.history["val_accuracy"],color="r",label="Testing Accuracy")
ax2.set_xticks(np.arange(1,25,1))
ax2.title.set_text("ACCURACY")

legend = plt.legend(loc="best",shadow=True)
plt.tight_layout()
plt.show()




values = [
"safe driving",
"texting - right",
"talking on the phone - right",
"texting - left",
"talking on the phone - left",
"operating the radio",
"drinking",
"reaching behind",
"hair and makeup",
"talking to passenger"
]





def make_prediction(img):
    img = image.load_img(img,target_size=(64,64))
    img = image.img_to_array(img)
    result = model.predict((np.expand_dims(img,axis=0)/255 - 0.5))[0] > 0.5
    return values[result.tolist().index(True)]
    
    
    
    
    
# test
import os # to dealing with Images folders in Train and Test 
import numpy as np # to dealing with arrays to convert Images To Array 
import pandas as pd # To dealing with csv files 
import matplotlib.pyplot as plt # To Plotting Loss and Accuracy and Show Images On Test Model 
from tensorflow.keras.models import Sequential # The Model We Will Use in our project 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint # to save the model in the best case of accuracy 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout , Dense,Flatten # Model Layers 
from tensorflow.keras.models import load_model

values = [
"safe driving",
"texting - right",
"talking on the phone - right",
"texting - left",
"talking on the phone - left",
"operating the radio",
"drinking",
"reaching behind",
"hair and makeup",
"talking to passenger"
]
trained_model = load_model("model.h5")
trained_model.compile(optimizer = 'adam',loss="categorical_crossentropy",metrics=['accuracy'])

def make_prediction(img):
    img = image.load_img(img,target_size=(64,64))
    img = image.img_to_array(img)
    result = trained_model.predict((np.expand_dims(img,axis=0)/255 - 0.5))[0] > 0.5
    return values[result.tolist().index(True)]

print(make_prediction("imgs/test/img_100018.jpg"))
print(make_prediction("imgs/test/img_1.jpg"))
 

