### Introduction
This project tries to predict pedestrian's trajectory using simple transformers and we combine the regression and classifcation loss to conduct multimodal prediction.
Pedestrian trajectory prediction task is a typical task of motion prediction. Generally, the model needs to encode the trajectory sequence and context information (social interaction) to predict the future motion. Both optimization method and data-driven method are applied to the task. Most recent successes on forecasting the people motion are based on LSTM models[1]. It is natural for researchers to think about replacing the LSTM model with transformers which generally outperform LSTM in modeling sequences. Here, we revise and train a simple transformer model from [2] as pedestrain trajectory predictor.

### Model selection
The trajectory prediction should be a multimodal problem by nature. Some works use mixture density network to directly conduct multiple regression. However, it often suffers mode collapse. Currently, the variation autoencoder and two-stage (first get rough class and then do regression) are commonly used approach to predict multiple possible future trajectories. The VAE method can map the input into some latent spaces which have meaningful representation of driving intention or styles. The two stage method can directly divided the output space (trajectory or region) into classes, which is easier to implement. In this study, we choose to use the latter.

### Method
The model is similar to the original transformer:  the embedding length is 512; it has 6-layer encoder and decoder with 8 attention heads.  we masked certain trajectory positions and the model will learn to predict it.

In our model, the input is the history two-dimensional Cartesian coordinates and we use a linear function to project them into the high dimentional embedding space. The position encoding uses trigonometric periodic functions to represent trajectory points' relative positions.

It is always helpful to encode the context and social interaction of the agents, but we don't contain these information here since we need additional feature extractor such as GCN to do this. 

To train the regression model, we directly use the L2 loss between the the predicted coordinates and the annotated positions. And for the classification tasks, we first cluster the trajectories into 1000 classes (potential anchors) represented as one-hot vectors. To train the multi-task models, we use the weighted sum of the L2 loss and cross-entropy loss as final loss function.
![image](https://github.com/jrcblue/cs496prejrc.github.io/blob/gh-pages/images/eth.PNG)
### Dataset

We use the ETH pedestrain trajectory dataset[3] as our training and test dataset. There are 12298 trajectories extracted from bird-eye-view images as shown in the figure. The input is the 8 pairs of coordinates in the past 3.2 seconds and the groundtruth is the 12 pairs in the future 4.8 seconds.  

### Experiments

We trained three different models including classification model, unimodal regression model and the regression + multimodal classification model. The training processes take around 20 epoches (~an hour) since the input and optimization object are simple. The average displacement error (ADE) and final displacement error (FDE) are commonly used to measure the prediction results (The smaller they are, the more accurate the prediction is). The results are shown in the follows (also compared to the BERT-based model in the paper). We could also put the trajectories onto the original bird-eye-view images[2]

The model combining the regression and multimodal classification performs better in terms of final displacement errors. However, all of the results are not as good as the LSTM ones. The possible reasons are that the SOTA LSTM models encode the context information and also our naive transformer is not specifically optimized for the motion prediction. 

### Conclusion and Future Plan
This study shows the transformer can model the trajectries sequence well but I don't think this is a very efficient way to conduct multimodal predictions. And we also find most of the results are concentrated in a certain regions. Recently, some work[4] use region proposals and transformers to give prediction in different areas, and some works apply VAE to model the latent spaces as driver's intention.

For the next step, I plan to use the transformer as trajectory encoder anc decoder and utilize the VAE model to conduct multimodal prediction. What's interesting is that we can build multiple latent space to model different attributes of agents (e.g. driving styles, intentions, classes of different vehicles). This can give us elaborate control of the prediction
 




