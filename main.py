import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Perceptron:
    
    def _init_(self, lr, n_epochs):
        # Construtor, define taxa de aprendizado e número máximo de épocas
        self.lr = lr
        self.n_epochs = n_epochs
        self.bias = -1
    def activation(self, value):
        # 1 se value >= 0, -1 se value < 0,  
        return (1 if value >= 0 else -1)
        
    def predict(self, x):
        # Multiplicação matricial entre as entradas e os pesos somado ao bias proporcional
        return np.dot(x, self.weights.T) + self.bias * self.w_bias
    
    def evaluate(self, target, predicted):
        # Calcula a diferença entre o valor real e o valor predito
        return (target - predicted)
    
    def train(self, x, d):
        # Definir aleatoriamente os pesos, o bias e o peso do bias
        # Enquanto houver erro, ou o máximo de épocas não for atingido continua o processo
            
        self.weights = np.random.random(x.shape[1])
        #print(x.shape)
        #self.bias = np.random.random()
        self.w_bias = np.random.random()
        #print(self.w_bias)
        
        epoch = 0
        is_error = True
        self.total_error = []
        
        while is_error and epoch < self.n_epochs:
            
            is_error  = False
            epoch_errors = 0
            
            # Para cada amostra
            for xi, target in zip(x, d):
                
                predicted = self.predict(xi)
                predicted = self.activation(predicted)
                
                current_error = self.evaluate(target, predicted)
                epoch_errors += current_error
                #print('epoch_error: ' + str(epoch_errors))
                
                # Se houve erro, atualizar os pesos
                if predicted != target:
                    self.weights += self.lr * current_error * xi
                    self.w_bias += self.lr * current_error * self.bias
                    is_error = True
                    #print(self.weights)
                    #print(self.w_bias)
                    
            self.total_error.append(epoch_errors/len(x))
            epoch += 1
            print('epoch: ' + str(epoch))
            
    def test(self, x):
        ''' Dado uma lista de X, submete-os à rede'''
        results = []
        
        if len(x.shape) != 1:
            for xi in x:
                predict = self.predict(xi)
                predict = self.activation(predict)
                results.append(predict)
        else:
            predict = self.predict(x)
            predict = self.activation(predict)
            results.append(predict)      
            
        return results


# Alterar conexão
engine = create_engine("mysql+pymysql://nome_usuario:senha@conexao:porta/ag002")
conn = engine.connect()
df = pd.read_sql_query('SELECT * FROM `breast-cancer`', con=conn)

#dropando o id para fazer o treino corretamente
df = df.drop("id", axis = 1)

#aleatoriezando os dados pegos no banco de dados
df= df.sample(frac=1).reset_index(drop=True)

#separando as classes das informacoes
results = df['class']
caracteristicas = df.drop('class', axis=1)

#quantidade de dados
lineNumber = len(df.index)

#dividindo as caracteristicas entre as de treino e as de teste
caracTrain = caracteristicas.head(int((lineNumber*0.8)))
caracTest = caracteristicas.tail(int(lineNumber*0.2 + 1))

#dividindo as classes entre as de treino e as de teste
resultsTrain = results.head(int((lineNumber*0.8)))
resultsTest = results.tail(int(lineNumber*0.2 + 1))

#formatando dados  para numpy array
caracTrainNp = np.array(caracTrain)
caracTestNp = np.array(caracTest)

resultTrainNp = np.array(resultsTrain)
resultTestNp = np.array(resultsTest)

#encaixando dados na funcao havesize simetrica
resultTrainNp[resultTrainNp == 1] = -1
resultTrainNp[resultTrainNp == 2] = 1

resultTestNp[resultTestNp == 1] = -1
resultTestNp[resultTestNp == 2] = 1


#fazendo treino da rede (taixa de aprendizado de 0.0005, e numero de epocas = 750000)
p = Perceptron(lr = 0.005, n_epochs = 100)
p.train(x = caracTrainNp, d = resultTrainNp)

teste_resultado = p.test(caracTestNp)
print(teste_resultado)

#criando uma matrix de confução e printando no termial
matrix = confusion_matrix(resultTestNp, teste_resultado)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()



while True:

    saida = input('Deseja entrar com um novo exame? (s/n)')
    if saida.lower() == 'n':
        break
    elif saida.lower() == 's':
        age = input('entre com o age: ')
        menopause = input('entre com o menopause: ') 
        tumorsize = input('entre com o tumeor size: ')
        invnodes = input('entre com o inv node: ')
        nodecaps = input('entre com o nome caps: ')
        degmalig = input('entre com o deg malisg: ')
        breast = input('entre com o breast: ')
        breastquad  = input('entre com o breast quad: ')
        irradiat = input('entre com o irradia: ') 

        newTest = np.array([age, menopause, tumorsize, invnodes, nodecaps, degmalig, breast, breastquad, irradiat]).astype(np.int32)
        returnNewTest = p.test(newTest)

        print('chance de recorrencia da doença')
        if returnNewTest == 1:
            print('sim')
        else:
            print('não')
    else: 
        print('entrada invalida')
    