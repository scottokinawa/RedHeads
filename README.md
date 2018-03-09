# RedHeads

[See Code](https://github.com/scottokinawa/RedHeads/blob/master/CNN.ipynb)


I wanted to test how well a simple CNN compares to a more complex facial recognition library in distinguishing one face from another.

For the sake of this small project I thought it would be fun to compare my face amongst other red headed men. People seem to always confused redheads together and I was wondering if a CNN would too. I have usually seen simple CNN's classify dogs from cats or even certain breeds of dogs from each other; however is it adept enough to distinguish faces from another?

### My hypothesis:
- I thought it the CNN would be able to detect my face from other redheads but at an accuracy of 80% if that. That means that the CNN doesn't really come close to its facial recognition counter part in identifying faces. 

### How I went about my project: 
#### How I collected the data: 
I was able to scrape 1,300 pictures of redheaded men around the ages of 20-30. All the pictures had their faces visible and had different backgrounds. For my own pictures, I took every pictures off of facebook of myself I could grab. For the rest I took pictures of myself in different back grounds. I was able to collect 400 pictures of myself after many many selfies... It was my first time feeling like a full fledged mileniall taking so many selfies of myself haha.

#### CNN:
I created a simple CNN algorithm with three convolutional layers, a 50% drop out and augmentation. I ran the code on AWS.

I don’t want to get too deep into how a CNN works, but as seen above there is a convolutional layer, pooling layer, fully connected layer and output layer. This is very high level, as there are different activation functions, penalties and softmax functions involved as well.

I decided to use three convolutional and pooling layers in my model with ReLu as the activation function. I could have added more convolutional layers into my model, but I thought it would become too computationally heavy even for the GPU (AWS) I was using.

Given the unbalanced classes of having only 400 pictures of myself and 1,300 of other redheaded men I included a drop out rate of 50% to help aid this discrepancy.

#### Facial Recognition: 
I utilized a Face Recognition library built by ageitgey in which they used dlib to build. Supposedly, “The model has an accuracy of 99.38% on the Labeled Faces in the Wild benchmark.”

If you view Adam Geitgey’s Medium post it goes into depth on how this works.

-First, look at a picture and find all the faces in it
-Second, focus on each face and be able to understand that even if a face is turned in a weird direction or in bad lighting, it is still the same person.
-Third, be able to pick out unique features of the face that you can use to tell it apart from other people — like how big the eyes are, how long the face is, etc.
-Finally, compare the unique features of that face to all the people you already know to determine the person’s name.

When using it to identify myself apart from other redheads, it worked remarkably well. Just for fun I tried to use it on my dog, but that didn’t work. It also didn’t work when using it on older pictures of myself when I was 12 years old. Of course that gets more complicated even for the human eye.

### Pitfalls of my approach:
The biggest pitfalls to my approach are not having enough pictures and imbalanced data.

Why are these possible issues?

CNN’s need a lot of pictures to work well. Many times CNN models use millions of pictures, of course this is not always necessary, but leads to better results.
Imbalanced data can lead to over fitting which is not good. For instance if you have 90% of your data in class 1 and 10% in class 2, then the model will just start assuming everything is class 1. To aid this process I did bring in drop out as a penalty measure, but it could have been beneficial to bring in synthetic sampling or just take more pictures of myself.

### Results:
My CNN model did work very well leading to an accuracy of 94%! However, as discussed before some pitfalls may have boosted this accuracy. When I used ageitgey’s Face Detection library, it was able to recognize me every time, however I only tried 50 times. Given the 99.38% accuracy found by Adam Geitgey, it would make obvious sense to use Face Recognition when differentiating human faces. However, simple CNN’s do seem to work rather well at differentiating me amongst my fellow red headed men. This small project lead to interesting results and I am glad A.I. can differentiate me from the other red heads when strangers can’t :)!
