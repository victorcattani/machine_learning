import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import uniform




#### Leitura do Data Set e separação dos dados de treino e teste ################
data_set=pd.read_csv("monthly-sunspots.csv", sep=";")
total_data_set = data_set.iloc[0:len(data_set),2].to_numpy()
y_training = data_set.iloc[0:len(data_set)-120,2].to_numpy()
y_test = data_set.iloc[(len(data_set)-120):,2].to_numpy()
months = np.array(range(np.shape(total_data_set)[0]))



#################### DETERMINAÇÃO DE FUNÇÕES NECESSÁRIAS AOS CÁLCULOS DAS SÉRIE TEMPORAIS ###########################################
### Função que adequa a matriz de entrada e de saída de acordo com o número de atrasos K ##############
def create_input_data (initial_train_data_set, K):
    train_data_size=len(initial_train_data_set)
    initial_X_train=np.zeros((train_data_size-K, K))
    initial_Y_train=np.zeros(train_data_size-K)
    
    for i in range(train_data_size-K):
        for j in range(K):
            initial_X_train[i, j] = initial_train_data_set[i+K-j-1]
            initial_Y_train[i] = initial_train_data_set[i+K]

    return initial_X_train, initial_Y_train
    
#### Função para a criação do k-fold #########################
def create_kfold (k,initial_X_train,initial_Y_train,i):
    len_X_training = len(initial_X_train)
    n= len_X_training/k
    training_indices = np.arange(0, len_X_training)
    if i ==0:
            validation_indices = np.arange(0,n)
    else:
            validation_indices = np.arange(i*n,(i+1)*n)
                            
    validation_indices=validation_indices.astype(int)
    train_indices = np.delete(training_indices, validation_indices)
                    
    X_train = initial_X_train[train_indices]
    Y_train = initial_Y_train[train_indices]
                    
    X_validation = initial_X_train[validation_indices]
    Y_validation = initial_Y_train[validation_indices]
    
    return X_train, Y_train, X_validation, Y_validation

##### Determinação dos pesos atraés da equação normal ##########################  
def normal_equation (X_train, Y_train, param_reg):
    I_line = np.eye(X_train.shape[1])
    I_line[0,0]=0    
    X_trans = X_train.T
    reg_term = param_reg*I_line
    w=np.dot(inv(np.dot(X_trans, X_train)+reg_term), np.dot(X_trans, Y_train))
     
    return w

####### Erro quadrático médio e valores preditos pelo modelo proposto ################################################
def mean_squared_error (Y_data,y_hat):
    e2=np.square(y_hat-Y_data)
    MSE = (1/(len(Y_data)))*np.sum(e2)
    
    return MSE

def inver_norm (data,min_value, interval):
    final_value=((interval)*data)+min_value
    return final_value

######## Exercício 1 - Preditor Linear #################################################################################
k=4
lista_RMSE=[]
for K in range (1,25):
    lista=[]
    initial_X_train, initial_Y_train = create_input_data (y_training, K)
    for i in range (k):
       X_train, Y_train, X_validation, Y_validation = create_kfold(k,initial_X_train,initial_Y_train,i)
        
       X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)
       X_validation = np.concatenate((np.ones((len(X_validation), 1)), X_validation), axis=1)
     
       X_trans = X_train.T
       w = normal_equation(X_train, Y_train,0)
          
       y_hat = np.dot(X_validation,w)       
       MSE = mean_squared_error(Y_validation, y_hat)
   
       lista.append(MSE)
    lista_RMSE.append((np.sum(lista)/k)**0.5)

fig_1A=plt.figure(figsize=(10,6))
plt.plot(lista_RMSE ,color="black", linewidth=3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel("Número de Entradas (K)", fontsize=18)
plt.ylabel("RMSE",fontsize=18)
fig_1A.savefig('RMSE_dados_treino_Ex_1')



K_min = np.argmin(lista_RMSE) + 1
print(K_min)


#### Determinação dos valores de W a partir do Hiperparâmetro  #######

X_train_final, Y_train_final = create_input_data (y_training, K_min)
vetor_bias=np.ones((len(X_train_final),1))
X_train_final = np.concatenate((vetor_bias, X_train_final), axis=1)

X_train_final_trans = X_train_final.T

w_teste = normal_equation(X_train_final, Y_train_final,0)


### Comparativo entre os dados de teste e o dados do preditor linear ######

X_teste, Y_teste = create_input_data (y_test, K_min)
vetor_bias=np.ones((len(X_teste),1))
X_teste = np.concatenate((vetor_bias, X_teste), axis=1) 

y_hat_teste=np.dot(X_teste,w_teste)

MSE_teste = mean_squared_error (y_hat_teste, Y_teste)


fig1B=plt.figure(figsize=(10,6))
plt.plot(months[len(months)-(120-K_min):len(months)],Y_teste ,color="blue", label="Real", linewidth=3, marker="o")
plt.plot(months[len(months)-(120-K_min):len(months)],y_hat_teste, color="red", label="Predito", linewidth=3, marker="s")
plt.legend(fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel("Mês", fontsize=18)
plt.ylabel("Número de Manchas Solares",fontsize=18)
fig1B.savefig("PreditoXReal_Modelo Linear")

print(MSE_teste**0.5)


############ Exercício 2 - Preditor Linear que usa como dado de entrada predições não Lineares ##################################
K_entrada = 8
k_fold=4
n_param_reg=4
T_lim=100
matriz_MSE_final = np.zeros((T_lim,n_param_reg))


min_data = min(total_data_set.astype(np.float))
data_interval = max(total_data_set.astype(np.float))-min_data
norm_data_set = (total_data_set.astype(np.float)-min_data)/data_interval
train_data_norm = norm_data_set[0:len(data_set)-120]
test_data_norm = norm_data_set[(len(data_set)-120):]
w_k = uniform(0, 0.3, size=(K_entrada, T_lim))

X_train_data_norm, Y_train_data_norm = create_input_data (train_data_norm, K_entrada)
X_test_data_norm, Y_test_data_norm = create_input_data (test_data_norm, K_entrada)

final_X_train_data=np.tanh(np.dot(X_train_data_norm,w_k))
final_X_test_data =np.tanh(np.dot(X_test_data_norm,w_k))



for T in range(1,T_lim+1):
    contador=0
    X_train=final_X_train_data[:,0:T] 
    for param_reg in np.geomspace(2,16,4):
        contador+=1
        lista_MSE=[]
        for i in range(k_fold):
            final_X_train, final_Y_train, final_X_validation, final_Y_validation = create_kfold(k_fold,X_train,Y_train_data_norm, i)
            
            final_X_train = np.concatenate((np.ones((len(final_X_train), 1)), final_X_train), axis=1)
            final_X_validation = np.concatenate((np.ones((len(final_X_validation), 1)), final_X_validation), axis=1)
               
            w=normal_equation(final_X_train,final_Y_train, param_reg)
            
            y_hat = np.dot(final_X_validation, w)
            y_hat_desnorm = inver_norm(y_hat,min_data,data_interval)
            final_Y_validation_desnorm=inver_norm(final_Y_validation,min_data,data_interval)
            
            MSE = mean_squared_error (final_Y_validation_desnorm, y_hat_desnorm) 
            lista_MSE.append(MSE)
            
        matriz_MSE_final[T-1,contador-1]=(1/k_fold)*np.sum(lista_MSE)
        
rms_K_val = np.sqrt(matriz_MSE_final) 


matriz_min=np.zeros(rms_K_val.shape[0])
for i in range (len(rms_K_val)):
    matriz_min[i]=np.min(rms_K_val[i,:]) 
                 
    
fig_A=plt.figure(figsize=(10,6))
plt.plot(matriz_min ,color="black", linewidth=3)
plt.legend(fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel("T", fontsize=18)
plt.ylabel("RMSE",fontsize=18)
fig_A.savefig('Exercício_2_RMSE_Treino')

#
#######################################################################################################################
###### Determinação de W a partir do valor de T e Lambda escolhido#####################################################
#######################################################################################################################

lista_lambda=[]
for i in (np.argmin(rms_K_val, axis=1)):
    A=np.geomspace(2,16,4)
    lista_lambda.append(A[i])
    
fig_B=plt.figure(figsize=(10,6))
plt.plot(lista_lambda ,color="blue", linewidth=3)
plt.legend(fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel("T", fontsize=18)
plt.ylabel("Lambda",fontsize=18)
fig_B.savefig('Lambda X T')

T_otm=np.argmin(matriz_min)+1
lambda_otm=lista_lambda[T_otm-1]

NL_train_data = final_X_train_data[:,0:T_otm]
vetor_bias=np.ones((len(NL_train_data),1))
NL_train_data = np.concatenate((vetor_bias, NL_train_data), axis=1)

w_train = normal_equation (NL_train_data,Y_train_data_norm,lambda_otm)

y_hat_train=np.dot(NL_train_data,w_train)
y_hat_desnorm_train = inver_norm(y_hat_train,min_data,data_interval)
NL_Y_train_desnorm=inver_norm(Y_train_data_norm,min_data,data_interval)

#plt.plot(NL_Y_train_desnorm, color="blue")
#plt.plot(y_hat_desnorm_train, color="red")

RMSE=np.sqrt(mean_squared_error (y_hat_desnorm_train, NL_Y_train_desnorm))

######### Normalização dos dados de Teste e determinação do y_hat e RMSE de teste ############## 
NL_test_data = final_X_test_data[:,0:T_otm]

vetor_bias=np.ones((len(NL_test_data),1))
NL_test_data = np.concatenate((vetor_bias, NL_test_data), axis=1) 

y_hat_teste=np.dot(NL_test_data,w_train)

y_hat_teste_desnorm = inver_norm(y_hat_teste, min_data, data_interval)
Y_test_data_desnorm = inver_norm(Y_test_data_norm, min_data, data_interval)

fig=plt.figure(figsize=(10,6))
plt.plot(months[len(months)-(120-K_entrada):len(months)],Y_test_data_desnorm ,color="blue", label="Real", linewidth=3, marker="o")
plt.plot(months[len(months)-(120-K_entrada):len(months)],y_hat_teste_desnorm, color="red", label="Predito", linewidth=3, marker="s")
plt.legend(fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel("Mês", fontsize=18)
plt.ylabel("Número de Manchas Solares",fontsize=18)
fig.savefig("PreditoXReal_Modelo não linear")

RMSE_teste=np.sqrt(mean_squared_error (y_hat_teste_desnorm, Y_test_data_desnorm))
print(RMSE_teste)
