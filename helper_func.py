import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

############################################################################################################################
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    if(len(im_data.shape)==2):
        height , width = im_data.shape
    else:
        height , width , depth = im_data.shape
    
    # what size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi) , height / float(dpi)
    
    # create a figure of the right size with one axes that takes up the ull figure 
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    
    # Hide spines, ticks etc.
    ax.axis('off')
    
    #display the image
    ax.imshow(im_data , cmap="gray")
    plt.show()
    
###############################################################################################################################
###############################################################################################################################
def create_bounding_boxes(image_url):
    ''' return: array of img_url string created by bounding boxes... '''
    
    img_obj_list = []
    
    image = cv2.imread(image_url)
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     print("gray")
#     cv2.imwrite("bbox_gray.jpg",gray)
#     display("bbox_gray.jpg")
    blur = cv2.GaussianBlur(gray,(7,7),0)        #object,size of blurring,
#     print("blur")
#     cv2.imwrite("bbox_blur.jpg",blur)
#     display("bbox_blur.jpg")
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]       # blurr object , range ,
#     print("thresh")
#     cv2.imwrite("bbox_thresh.jpg",thresh)
#     display("bbox_thresh.jpg")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,13))
    
    dilate = cv2.dilate(thresh, kernel , iterations=1)
#     print("dilate")
#     cv2.imwrite("bbox_dilate.jpg",dilate)
#     display("bbox_dilate.jpg")
#     cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    
    cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])
    i=1
    word_bounding_boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h<30:
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),1)
            word_bounding_boxes.append((x,y,w,h))
            
            # Crop the region from the image
            cropped_image = image[y:y+h, x:x+w]           
            
#             print(f"bbox{i}")
            
            
            # Save the cropped image
            cv2.imwrite(f'words/cropped_image{i}.jpg', cropped_image)
#             display(f'words/cropped_image{i}.jpg')
            
            img_obj_list.append(str(f'words/cropped_image{i}.jpg'))
            
            i+=1
               
    return img_obj_list



###############################################################################################################################
###############################################################################################################################

def detect_letter(model:nn.Module,
                 img_url:str,
                 class_names:list):
    
    '''returns : detected_later in string '''
    
    # Load the thresholded image
    thresh_img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)

    # Define the desired size
    desired_size = 28

    # Find the current size
    height, width = thresh_img.shape

    # Calculate border sizes
    border_vertical = (desired_size - height) // 2
    border_horizontal = (desired_size - width) // 2

#     # Print border values and input array dimensions
#     print("Border values:", top, bottom, left, right)
#     print("Input array dimensions:", _src.ndim)

    if (border_vertical>=0 and border_horizontal>=0):
        # Add borders to the image
        bordered_img = cv2.copyMakeBorder(thresh_img, border_vertical, border_vertical, border_horizontal, border_horizontal, cv2.BORDER_CONSTANT, value=0)

    else:
        bordered_img = thresh_img
    # Resize the image to desired size
    resized_img = cv2.resize(bordered_img, (desired_size, desired_size))


    #converting numpy to tensor 
    resized_image_tensor = torch.from_numpy(resized_img)

#     type(resized_image_tensor),resized_image_tensor.shape



    sample = resized_image_tensor
    model.eval()
    with torch.inference_mode():
        sample = sample.type(torch.float)
        sample = torch.unsqueeze(sample,dim=0)
        sample = torch.unsqueeze(sample,dim=0)

        pred_logit = model(sample)

        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

        pred_class = pred_prob.argmax(dim=0)

#         print(class_names[pred_class])
    return str(class_names[pred_class])

###############################################################################################################################
###############################################################################################################################
import cv2
def words_to_text(model:nn.Module,img_url:str,class_names:list):
    
    text = ""
    
    img = cv2.imread(img_url)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    x , y = thresh.shape
    
    ##################### Image Refactoring ###################
    x_new = 28
    y_new = math.ceil((x_new/x) * y)
    ###########################################################
    resized_image = cv2.resize(thresh, (y_new,x_new))
    
    
    cnts = cv2.findContours(resized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]

    cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])
    i=1
    
    word_bounding_boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h<30:
            cv2.rectangle(resized_image,(x,y),(x+w,y+h),(36,255,12),1)
            word_bounding_boxes.append((x,y,w,h))

            # Crop the region from the image
            cropped_image = resized_image[y:y+h, x:x+w]           

#             print(f"bbox{i}")
            cv2.imwrite(f'words/i{i}.jpg', cropped_image)
            txt = detect_letter(model=model,img_url=str(f'words/i{i}.jpg'),class_names=class_names)
            
            text+=txt
            i+=1
    
#     print(text)
    return text
     