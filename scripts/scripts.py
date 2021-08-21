# -*- coding: utf-8 -*-



def pic_to_vid(imagelist,fps):
    h,w,c = imagelist[0].shape
    size = (w,h) 
    out = cv2.VideoWriter("Video.avi",cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    for img in imagelist:
        out.write(img)
    out.release()   


def see_one_channel(single_channel):
    zero1 = np.zeros((256,512,2),dtype = np.uint8 )
    final = (np.concatenate((np.expand_dims(single_channel,axis =-1),zero1),axis = 2))*255
    cv2.imshow("window_name", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


def showimage(img,mode = None,label = None ):
    if mode != None:
        img = cv2.imread(img,-1)
        
    
    if label == None:
        img = np.squeeze(img)
        img = img*255
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("window_name", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()