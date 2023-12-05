from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import shutil
import numpy as np
import cv2

# Input the necessary path
source_folder = '/home/r15m/Desktop/Vietnamese-license-plate-recognition-main'               # Folder contains input image and two model
input_media = 'test_4.mp4'                                                                   # Name of image or video in source folder

# Show license plate function
def license_plate_show(labels_path):
  data = []
  labels = []
  x = []
  y = []
  x_above = []
  x_below = []
  labels_order = []
  characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

  def get_row_position(predict_table,y):
      postion = np.where(y==predict_table)
      if postion[0].size>0:
          return postion[0][0]
      else: return -1
  def get_value_from_table(predict_table,row_array,col):
      #col = 0 <-> labels ; col = 1 <-> x; col = 2 <-> y
      x_array = []
      for num in row_array:
          pos = get_row_position(predict_table,num)
          x_array.append(predict_table[pos][col])
      return x_array
  # Open txt file to get x,y and labels values
  with open(labels_path,'r') as file:
      lines = file.readlines()
      for line in lines:
          line = line.strip().split(',')
          for data in line:
              data = data.split(' ')
              labels.append(data[0])
              labels = [int(num) for num in labels]
              x.append(data[1])
              x = [float(num) for num in x]
              y.append(data[2])
              y = [float(num) for num in y]
  predict_table = np.column_stack((labels,x,y))

  # Locate y for characters in row above and below
  y.sort()
  row_below = []
  row_above = []
  c = y[0]
  for num in y:
      if (num>1.8*c):
          row_below.append(num)
      else:
          row_above.append(num)

  # Get x corresponding to y value
  x_above = get_value_from_table(predict_table,row_above,1)
  x_below = get_value_from_table(predict_table,row_below,1)
  x_above.sort()
  x_below.sort()
  # Get the order of characters in plates
  labels_order = get_value_from_table(predict_table,x_above,0)
  labels = get_value_from_table(predict_table,x_below,0)
  for num in labels:
      labels_order.append(num)

  # Show the license plates
  license_plate = ''
  for i in range(len(labels_order)):
      for j in range(len(characters)):
          if labels_order[i] == j:
              license_plate += characters[j]
      if row_above != [] and row_below != [] and np.shape(row_above)== (3,):
          if(i==2): license_plate += '\n'
          elif(i==5): license_plate +='.'
      elif row_above != [] and row_below != [] and np.shape(row_above) !=(3,):
          if(i==1): license_plate += '-'
          elif(i==3): license_plate += '\n'
          elif(i==6): license_plate +='.'
      elif row_above == [] or row_below == []:
          if(i==2): license_plate += '-'
          elif(i==5): license_plate +='.'
  return license_plate

# Show license plate on image function
def put_plate_on_image(img,source_path,detected_label,license_plate_predict):
    list_image = os.listdir(detected_label)
    label_path = os.path.join(detected_label,list_image[0])
    def show_plate_on_image(contour_images_path,contour_labels_path):
        image = cv2.imread(contour_images_path)
        with open(contour_labels_path,'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(',')
                for data in line:
                    data = data.split(' ')
        data = [float(num) for num in data]
        data = data[1:]
        return data
    contour_data = show_plate_on_image(source_path,label_path)
    height, width, channels = np.shape(img)
    x = width*contour_data[0]
    y = height*contour_data[1]
    w = width*contour_data[2]
    h = height*contour_data[3]
    x_show= int(x - w/2)
    y_show = int(y + h)
    text = license_plate_predict
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 3)
    line_height = text_size[1] + 40
    for i, line in enumerate(text.split("\n")):
        y_scale = y_show + i * line_height
        cv2.putText(img,line,(x_show,y_scale), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3, 2)
    return img

# Initialize camera
video_path = source_folder + '/' + input_media # get 1 file video from source folder
cap  = cv2.VideoCapture(video_path)

 # Input path
source_path = os.path.join(source_folder,'image.jpg')
detected_label = os.path.join(source_folder,'results/labels')
path_rec = os.path.join(source_folder,'results/rec_results/labels')
model_detect = YOLO(source_folder + '/' + 'license_plate_detection/weights/last.pt')
model_rec = YOLO(source_folder + '/' + 'characters_recognition/weights/last.pt')

# Loop
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret:
        # Remove the previous results
        if os.path.exists(source_folder + '/results'):
            shutil.rmtree(source_folder + '/results')
        frame = cv2.resize(frame,(640,640))
        
        # Detect the license plates
        results_detect = model_detect(frame, conf = 0.4, save=True, save_txt = True, save_crop = True, project = source_folder, name = 'results')
        
        # Recognize the plate's characters
        for result in results_detect:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = frame[r[1]:r[3], r[0]:r[2]]
                results_rec = model_rec(crop, conf = 0.4, save=True, save_txt = True, project = source_folder + '/results', name = 'rec_results')
                path_labels = os.listdir(path_rec)
                for i in range(len(path_labels)):
                    l_image = os.path.join(path_rec,path_labels[i])
                    plate_predict = license_plate_show(l_image)
                    print(plate_predict)
                    predict_image = put_plate_on_image(frame,source_path,detected_label,plate_predict)
                    cv2.imshow("Plate crop",crop)
        annotated_frame = results_detect[0].plot()
        
        # Display the annotated frame
        cv2.imshow("Predict the license plate", annotated_frame)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit show mode
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
