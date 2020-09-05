import  torch
import numpy as np
#inner model layers are given by list(model._modules['features'])
#and the classifier layer is given by model._modules['classifier']
import experiment
import utils
import matplotlib.pyplot as plt
import random






#The following function returns the
def Peturbutation(X,relevance_score,num_most_relevant_features):

    #relevance_score=torch.abs(relevance_score)
    indexes = torch.topk(relevance_score, num_most_relevant_features)[1]
    X_=X.clone()
    X_[0,:][indexes]=random.randint(1,123468)#Instead the relevant word we put a random word from
    #the vocabulary
    return X_



#lrp function gets the input X and list of all layers in the net _layers
#the function returns the relavance score for features in input X
#In our case X is the embedding representation of document tokens.
def Lrp(X,_layers,_L):
    A = [X] + [None] * _L  # initiliza the values of the activation layers
    for l in range(_L):
        if l == 2:
            A[l] = A[l].view((A[l].shape[0], -1))#we fit the second shape
        A[l + 1] = _layers[l].forward(A[l])  # we fill the activation values of matrix
    chosen_label = A[_L].argmax().item()
    T = torch.FloatTensor(
        (1.0 * (np.arange(20) == chosen_label)))  # \\this is the mask/indicator which shows who is the true label.
    R = [None] * _L + [(A[-1] * T).data]  # we initialize the relavance by taking the last layer [(A[-1] * T).data]

    for l in range(1, _L)[::-1]:
        #the conditions holds for all layers except for the relu layer
        A[l] = (A[l].data).requires_grad_(True)
        #below incr and rho are help functions
        incr = lambda z: z + 1e-9
        rho = lambda p: p
        #Below we perform four steps for calculate the relevance scores for each layer.
        #The alorithm is iterative, we use the current relance scores layer to calculate the relevance scores
        #of the preavious layer
        z = incr(utils.newlayer(_layers[l], rho).forward(A[l]))  # step 1
        if l == 1:
            # print(R[l+1].view((1,30,-1)).shape,z.shape)
            R[l + 1] = R[l + 1].view((1, 30, -1))
        s = (R[l + 1] / z).data
       # print((z*s).sum())# )step 2
        (z * s).sum().backward()
        c = A[l].grad  # step 3
        R[l] = (A[l] * c).data  # step 4

    return torch.sum(R[1].view((400,300)),dim=1)#Recall the R[1] is the first layer relevance, R[0] is None ;400 and 300
#is the dimension of the chosen document' embedding.


#SA is another Intepretability algorithm. It also gets X(Document representation) and the list of layers
#of a trained net
def Sa(X,_layers,_L):
    A = [X] + [None] * _L  # initiliza the values of the activation layers
    for l in range(_L):
        if l == 2:
            A[l] = A[l].view((A[l].shape[0], -1))  # we fit the second shape
        A[l + 1] = _layers[l].forward(A[l])  # we fill the activation values of matrix
    chosen_label = A[_L].argmax().item()
    T = torch.FloatTensor(
        (1.0 * (np.arange(20) == chosen_label)))  # \\this is the mask/indicator which shows who is the true label.
    R = [None] * _L + [(A[-1] * T).data]  # we initialize the relavance by taking the last layer [(A[-1] * T).data]
    #SA calculation
    d = R[_L]
    #print(A)
    for l in range(1, _L)[::-1]:

        #if isinstance(_layers[l], torch.nn.Conv1d) or isinstance(_layers[l], torch.nn.Linear):

        A[l] = (A[l].data).requires_grad_(True)

        if l == 1:
            #A[l]=A[l].view((A[l].shape[0],-1))
            d = d.view((d.shape[0], 30,-1))  # we fit the second shape

        rho = lambda p: p
        z=utils.newlayer(_layers[l], rho).forward(A[l])

        #print(l, _layers[l])
        #print(A[l], d)
       #print(_layers[l])
        z.backward(d)
        d = A[l].grad
        #print(l, d.shape)


    return torch.sum(d.view(400, 300), dim=1)#we return the relevance score




if __name__ == "__main__":
    model = experiment.articlenet(300,123470)
    model.eval()
    model.load_state_dict(torch.load('model_advanced.pkl'))
    layers = [module for module in model.modules()][1:]#net of the layers list

    L = len(layers)#Number of the layers in the trained net

    Train_path = "Train_tokens"
    paths_list = Train_path
    word_dict = experiment.get_vocabs(Train_path)  # We create the vocabulary according the Trainset
    test = experiment.Dataset_model(word_dict, subset='Test')#we run on train set
    train_dataloader = experiment.DataLoader(test, batch_size=1, shuffle=False)
    #we define list of tensor
    relevance_scores_lrp_list=[]
    relevance_scores_sa_list=[]
    accuracy_lrp=[]
    accuracy_sa=[]

    for i,data_input in enumerate(train_dataloader):
        document,true_label=data_input
        X=document
        #We calculate the relevance scorss for the words in the documents
        relevance_score_lrp=Lrp(X,layers,L)
        relevance_score_sa=Sa(X,layers,L)




        relevance_scores_lrp_list.append(relevance_score_lrp)#we save the relevance scores for lrp and for sa
        relevance_scores_sa_list.append(relevance_score_sa)


    #Now we perform the peturbutation:

    maxacc=0
    num_of_peturbutations=50
    for p in range(num_of_peturbutations):
        acc_lrp=0
        acc_sa=0
        for i,data_input in enumerate(train_dataloader):
            document,true_label=data_input
            X=document
            #We perform the pertubations
            X_peturbutated_lrp=Peturbutation(X,relevance_scores_lrp_list[i],p)
            X_peturbutated_sa = Peturbutation(X, relevance_scores_sa_list[i], p)

            output_lrp=model(X_peturbutated_lrp)
            predicted_lrp=torch.max(output_lrp.data, 1)[1]#we get the predicted label by the model
            #after each pertubations for both lrp and sa
            output_sa=model(X_peturbutated_sa)
            predicted_sa=torch.max(output_sa.data, 1)[1]
            if  predicted_lrp==true_label:
                acc_lrp+=1

            if predicted_sa == true_label:
                acc_sa += 1

        if p==0:
            maxacc_lrp=acc_lrp/len(test)
            maxacc_sa=acc_sa/len(test)
        #We normalize the accuracy by dividing it by the maximum accuracy
        accuracy_lrp.append((acc_lrp / len(test)/maxacc_lrp))

        accuracy_sa.append((acc_sa / len(test))/maxacc_lrp)


    #We plot the graph
    print(accuracy_lrp)
    print(accuracy_sa)


    plt.plot(accuracy_lrp,color='blue',label='LRP')
    plt.xlim(0,50)

    plt.plot(accuracy_sa,color='green',label='SA')
    plt.ylim(0,1)
    plt.xlabel('perturbations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()













