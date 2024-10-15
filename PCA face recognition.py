import cv2
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

image_1=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/1.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_1=cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_1)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()                                               # killing all the windows after work       

image_2=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/2.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_2=cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_2)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows() 

image_3=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/4.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_3=cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_3)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows() 
       
image_4=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/Newfaces/7.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_4=cv2.cvtColor(image_4,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_4)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()  

image_5=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/5.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_5=cv2.cvtColor(image_5,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_5)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()    

image_6=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/10.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_6=cv2.cvtColor(image_6,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_6)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()
    
image_7=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/9.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_7=cv2.cvtColor(image_7,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_7)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()

image_8=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/14.jpg",cv2.IMREAD_COLOR)   # reading an image 
image_8=cv2.cvtColor(image_8,cv2.COLOR_BGR2GRAY)                           # Converting color
cv2.imshow("image",image_8)                                             # showing the image
cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
cv2.destroyAllWindows()




image_1=np.reshape(image_1,((image_1.shape[0]**2),1))
image_2=np.reshape(image_2,((image_2.shape[0]**2),1))
image_3=np.reshape(image_3,((image_3.shape[0]**2),1))
image_4=np.reshape(image_4,((image_4.shape[0]**2),1))
image_5=np.reshape(image_5,((image_5.shape[0]**2),1))
image_6=np.reshape(image_6,((image_6.shape[0]**2),1))
image_7=np.reshape(image_7,((image_7.shape[0]**2),1))
image_8=np.reshape(image_8,((image_8.shape[0]**2),1))

mean_array=np.zeros(((image_1.shape[0]),1))
for i in range(image_1.shape[0]):
    a=image_1[i][0]/4
    b=image_2[i][0]/4
    c=image_3[i][0]/4
    d=image_4[i][0]/4

    mean_array[i][0]=a+b+c+d

mean_array=np.array(mean_array)

#difference between each image and mean
phi_1=np.zeros(((image_1.shape[0]),1))
phi_2=np.zeros(((image_2.shape[0]),1))
phi_3=np.zeros(((image_3.shape[0]),1))
phi_4=np.zeros(((image_4.shape[0]),1))

phi_1=image_1-mean_array
phi_2=image_2-mean_array
phi_3=image_3-mean_array
phi_4=image_4-mean_array

A=np.concatenate((phi_1,phi_2,phi_3,phi_4),axis=1)


A_transpose=A.transpose()
C=np.dot(A,A_transpose)
#M=np.array([1/4],dtype="float16")
C=(1/4)*C

eigenvalue,eigenvectors=lin.eig(C)

eigenvalue=eigenvalue.real
eigenvectors=eigenvectors.real


plt.plot(range(1,11),eigenvalue[0:10])


eigenvector_1=eigenvectors[0:eigenvectors.shape[1],0]
eigenvector_2=eigenvectors[0:eigenvectors.shape[1],1]
eigenvector_3=eigenvectors[0:eigenvectors.shape[1],2]
eigenvector_4=eigenvectors[0:eigenvectors.shape[1],3]


normalized_eigen_vector_value_1=np.linalg.norm(eigenvector_1)
normalized_eigen_vector_value_2=np.linalg.norm(eigenvector_2)
normalized_eigen_vector_value_3=np.linalg.norm(eigenvector_3)
normalized_eigen_vector_value_4=np.linalg.norm(eigenvector_4)

normalized_eigen_vector_1=(1/normalized_eigen_vector_value_1)*eigenvector_1
normalized_eigen_vector_2=(1/normalized_eigen_vector_value_2)*eigenvector_2
normalized_eigen_vector_3=(1/normalized_eigen_vector_value_3)*eigenvector_3
normalized_eigen_vector_4=(1/normalized_eigen_vector_value_4)*eigenvector_4





tnormalized_eigen_vector_1=np.reshape(normalized_eigen_vector_1,(1,normalized_eigen_vector_1.shape[0]))
tnormalized_eigen_vector_2=np.reshape(normalized_eigen_vector_2,(1,normalized_eigen_vector_1.shape[0]))
tnormalized_eigen_vector_3=np.reshape(normalized_eigen_vector_3,(1,normalized_eigen_vector_1.shape[0]))
tnormalized_eigen_vector_4=np.reshape(normalized_eigen_vector_4,(1,normalized_eigen_vector_1.shape[0]))

new_normalized_eigen_vector_1=np.reshape(normalized_eigen_vector_1,(normalized_eigen_vector_1.shape[0],1))
new_normalized_eigen_vector_2=np.reshape(normalized_eigen_vector_2,(normalized_eigen_vector_1.shape[0],1))
new_normalized_eigen_vector_3=np.reshape(normalized_eigen_vector_3,(normalized_eigen_vector_1.shape[0],1))
new_normalized_eigen_vector_4=np.reshape(normalized_eigen_vector_4,(normalized_eigen_vector_1.shape[0],1))

weigth_1_1=np.dot(tnormalized_eigen_vector_1,phi_1)
weight_2_1=np.dot(tnormalized_eigen_vector_2,phi_1)
weigth_3_1=np.dot(tnormalized_eigen_vector_3,phi_1)
weigth_4_1=np.dot(tnormalized_eigen_vector_4,phi_1)

weight_1_matrix=np.concatenate((weigth_1_1,weight_2_1,weigth_3_1,weigth_4_1),axis=1)

weigth_1_2=np.dot(tnormalized_eigen_vector_1,phi_2)
weight_2_2=np.dot(tnormalized_eigen_vector_2,phi_2)
weigth_3_2=np.dot(tnormalized_eigen_vector_3,phi_2)
weigth_4_2=np.dot(tnormalized_eigen_vector_4,phi_2)

weight_2_matrix=np.concatenate((weigth_1_2,weight_2_2,weigth_3_2,weigth_4_2),axis=1)

weigth_1_3=np.dot(tnormalized_eigen_vector_1,phi_3)
weight_2_3=np.dot(tnormalized_eigen_vector_2,phi_3)
weigth_3_3=np.dot(tnormalized_eigen_vector_3,phi_3)
weigth_4_3=np.dot(tnormalized_eigen_vector_4,phi_3)

weight_3_matrix=np.concatenate((weigth_1_3,weight_2_3,weigth_3_3,weigth_4_3),axis=1)

weigth_1_4=np.dot(tnormalized_eigen_vector_1,phi_4)
weight_2_4=np.dot(tnormalized_eigen_vector_2,phi_4)
weigth_3_4=np.dot(tnormalized_eigen_vector_3,phi_4)
weigth_4_4=np.dot(tnormalized_eigen_vector_4,phi_4)

weight_4_matrix=np.concatenate((weigth_1_4,weight_2_4,weigth_3_4,weigth_4_4),axis=1)


RImage1=mean_array+(new_normalized_eigen_vector_1*weigth_1_1)+(new_normalized_eigen_vector_2*weight_2_1)+(new_normalized_eigen_vector_3*weigth_3_1)+(new_normalized_eigen_vector_4*weigth_4_1)
RImage2=mean_array+(new_normalized_eigen_vector_1*weigth_1_2)+(new_normalized_eigen_vector_2*weight_2_2)+(new_normalized_eigen_vector_3*weigth_3_2)+(new_normalized_eigen_vector_4*weigth_4_2)
RImage3=mean_array+(new_normalized_eigen_vector_1*weigth_1_3)+(new_normalized_eigen_vector_2*weight_2_3)+(new_normalized_eigen_vector_3*weigth_3_3)+(new_normalized_eigen_vector_4*weigth_4_3)
RImage4=mean_array+(new_normalized_eigen_vector_1*weigth_1_4)+(new_normalized_eigen_vector_2*weight_2_4)+(new_normalized_eigen_vector_3*weigth_3_4)+(new_normalized_eigen_vector_4*weigth_4_4)


Error_1=np.linalg.norm(RImage1-image_1)
Error_2=np.linalg.norm(RImage2-image_2)
Error_3=np.linalg.norm(RImage3-image_3)
Error_4=np.linalg.norm(RImage4-image_4)

R1=np.reshape(RImage1,(100,100))
R1=np.array(R1,dtype="uint8")
cv2.imwrite("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/R1.jpg",R1)

R2=np.reshape(RImage2,(100,100))
R2=np.array(R1,dtype="uint8")
cv2.imwrite("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/R2.jpg",R2)

R3=np.reshape(RImage2,(100,100))
R3=np.array(R1,dtype="uint8")
cv2.imwrite("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/R3.jpg",R3)

R4=np.reshape(RImage2,(100,100))
R4=np.array(R1,dtype="uint8")
cv2.imwrite("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/R3.jpg",R4)


#New test image
image_test=cv2.imread("C:/Users/ESDM/Desktop/Face Recognition/NewFaces/2.jpg",cv2.IMREAD_COLOR)   # reading an image 
test_image=cv2.cvtColor(image_test,cv2.COLOR_BGR2GRAY)                           # Converting color
#cv2.imshow("image",test_image)                                             # showing the image
#cv2.waitKey(0)                                                        # Holding the windown untile user closes                  
#cv2.destroyAllWindows() 


        
test_image=np.reshape(test_image,((test_image.shape[0]**2),1))    
phi_test_image=test_image-mean_array

weigth_test_1=np.dot(tnormalized_eigen_vector_1,phi_test_image)
weight_test_2=np.dot(tnormalized_eigen_vector_2,phi_test_image)
weigth_test_3=np.dot(tnormalized_eigen_vector_3,phi_test_image)
weigth_test_4=np.dot(tnormalized_eigen_vector_4,phi_test_image)

weight_test_matrix=np.concatenate((weigth_test_1,weight_test_2,weigth_test_3,weigth_test_4),axis=1)


Error_test_1=np.linalg.norm(weight_1_matrix-weight_test_matrix)
Error_test_2=np.linalg.norm(weight_2_matrix-weight_test_matrix)
Error_test_3=np.linalg.norm(weight_3_matrix-weight_test_matrix)
Error_test_4=np.linalg.norm(weight_4_matrix-weight_test_matrix)

Error_test_list=[Error_test_1,Error_test_2,Error_test_3,Error_test_4]

for i in range(len(Error_test_list)):
    if round(Error_test_list[i],3)<(10):
        print("The test image belongs to",i+1,"category")
    elif(i==len(Error_test_list) and round(Error_test_list[i],3)>(10)):
        print("The test image does not belongs to any categories"