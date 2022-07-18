import numpy as np
import math
class RegLog():
    
    """
    
    Regresión Logística que usa el método de Newton-Raphson para optimizar sus parámetros.
    
    ...
    
    
    Métodos
    -------
    
    fit (x,y):
        Fittea el modelo con los x e y correspondientes.
    
    predict (x):
        Devuelve la predicción sobre un conjunto de datos x.
        
    coeficientes ():
        Devuelve los coeficientes de la regresión logística.
        
    predictProb (x):
        Devuelve la probabilidad de que cada dato de x pertenezca a la clase 1.
        
    """
    
    def __init__(self):
        """
        Constructor de la clase.
        """
        self.coef=[]

    def fit(self, x, y):
        """
        Fittea el modelo con los x e y correspondientes.
        
        Parámetros
        ------------
        
        x : array / list
            Conjunto de datos
            
        y : array / list
            Conjunto de etiquetas de los datos x
            
        """
        
        Filas = x.shape[0]
        Columnas = x.shape[1]
        
        # Defino H, que dado un dato y unos coeficientes devuelve la probabilidad de que el dato pertenezca a la clase 1.
        
        def H(x,coef):
            Aux = 0
            for i in range(Columnas+1):
                if i == 0:
                    Aux = Aux+coef[i]
                else:
                    Aux = Aux + x.iloc[i-1]*coef[i]
          
            return (1/(1 + np.exp(-Aux)))
           
        
        # Defino J (función de costo) que dado un conjunto x;y y unos coeficientes devuelve 
        # un "score" de cuan bien estos coeficientes ajustan el modelo.
        
        def J(x_train,y_train,coef):
            Aux2=0
            for j in range(Filas):
                Aux2 = Aux2 - (1/Filas)*(y_train[j]*math.log(H(x_train.iloc[j,:],coef),10) + (1-y_train[j])*math.log(1-H(x_train.iloc[j,:],coef),10))
            return Aux2
        
        # Defino Grad, que dado un conjunto x;y y unos coeficientes devuelve 
        # la dirección de máximo crecimiento de J en función de los coeficientes.
        
        def Grad(x_train,y_train,coef):
            
            LAux3=[]
            for j in range(Columnas+1):
                Aux3=0
                for i in range(Filas):
                    if j==0:
                        Aux3=Aux3-(1/Filas)*((y_train[i]-H(x_train.iloc[i,:],coef))/math.log(10))
                    else:
                        Aux3=Aux3-(1/Filas)*x_train.iloc[i,j-1]*((y_train[i]-H(x_train.iloc[i,:],coef))/math.log(10))
                LAux3.append(Aux3)
            return LAux3
                
        # Defino Grad, que dado un conjunto x;y y unos coeficientes devuelve 
        # el Hessiano de J, que es escencial para realizar el método de Newton - Raphson.
        
        def Hessiano(x_train,y_train,coef):
            LAux4=[]
            for k in range(Columnas+1):
                LAux4_1=[]
                for j in range(Columnas+1):
                    Aux4=0
                    for i in range(Filas):
                        if k==0 and j==0:
                            Aux4=Aux4-(1/Filas)*((H(x_train.iloc[i,:],coef)**2 - H(x_train.iloc[i,:],coef))/math.log(10))
                        elif k==0:
                            Aux4=Aux4-(1/Filas)*x_train.iloc[i,j-1]*((H(x_train.iloc[i,:],coef)**2 - H(x_train.iloc[i,:],coef))/math.log(10))
                        elif j ==0:
                            Aux4=Aux4-(1/Filas)*x_train.iloc[i,k-1]*((H(x_train.iloc[i,:],coef)**2 - H(x_train.iloc[i,:],coef))/math.log(10))
                        else:
                            Aux4=Aux4-(1/Filas)*x_train.iloc[i,j-1]*x_train.iloc[i,k-1]*((H(x_train.iloc[i,:],coef)**2 - H(x_train.iloc[i,:],coef))/math.log(10))
                    LAux4_1.append(Aux4)
                LAux4.append(LAux4_1)
            return LAux4
        
                    
                    
            
        
        # Creo la lista coef. Arranco con todos los coeficientes iguales a 0.
        coef=[]
        for i in range(Columnas+1):
            coef.append(0)
            
        # Método Newton Raphson.
                
        def Metodo_Newton_Raphson(x,y,coef):
            Aux5=np.array(np.linalg.inv(np.array(Hessiano(x,y,coef)))) @ Grad(x,y,coef)
            Coef_Nuevo=[]
            for i in range(Columnas+1):
                Coef_Nuevo.append(coef[i]-Aux5[i])
            Dist=0
            for i in range(Columnas+1):
                Dist=Dist + abs(Coef_Nuevo[i] - coef[i])**2
            Dist=Dist**0.5
            if Dist<0.00001:
                pass
            else:
                Metodo_Newton_Raphson(x,y,Coef_Nuevo)
            return Coef_Nuevo
            
        #Guardo los coeficientes para despues usarlos en el predict.
        
        self.coef=Metodo_Newton_Raphson(x,y,coef)

    def predict(self, x):
        """
        Devuelve la predicción sobre un conjunto de datos x.

        Parámetros
        ------------
        
        x : array / list
            Conjunto de datos a predecir

        Return
        -------
        
        list
            Predicción del modelo sobre el conjunto de datos x (0 o 1).
            
        """
        if len(self.coef) == 0 :
            
            raise ValueError ( "Todavía no se fiteó el modelo. Utilize el método fit para luego predecir.")
            
        else:   
            
            Filas = x.shape[0]
            Columnas = x.shape[1]
            def H(x,coef):
                Aux = 0
                for j in range(Columnas+1):
                    if j == 0:
                        Aux = Aux+coef[j]
                    else:
                        Aux = Aux + x.iloc[j-1]*coef[j]

                return (1/(1 + np.exp(-Aux)))

            predict=[]
            for i in range(Filas):
                if H(x.iloc[i,:],self.coef) >=0.5:
                    predict.append(1)
                else:
                    predict.append(0)
            return predict
    
    def coeficientes(self):
        """
         Devuelve los coeficientes de la regresión logística.
         
         Return
         -------
         
         list
             Coeficientes del modelo
        
        """
        
        if len(self.coef) == 0 :
            
            raise ValueError ( "Todavía no se fiteó el modelo. Utilize el método fit para encontrar los coeficientes.")
            
        else: 
            
            return self.coef
    
    
    
    def predictProb(self, x):
        
        """
         Devuelve la probabilidad de que cada dato de x pertenezca a la clase 1.
         
        Parámetros
        ------------
        
        x : array / list
            Conjunto de datos a predecir

        Return
        -------
        
        list
            Predicción del modelo sobre el conjunto de datos x (Probabilidad de que pertenezca a la clase 1).
        """
        
        if len(self.coef) == 0 :
            
            raise ValueError ( "Todavía no se fiteó el modelo. Utilize el método fit para luego predecir.")
            
        else:
        
            Filas = x.shape[0]
            Columnas = x.shape[1]
            def H(x,coef):
                Aux = 0
                for j in range(Columnas+1):
                    if j == 0:
                        Aux = Aux+coef[j]
                    else:
                        Aux = Aux + x.iloc[j-1]*coef[j]

                return (1/(1 + np.exp(-Aux)))

            predict=[]
            for i in range(Filas):
                predict.append(H(x.iloc[i,:],self.coef))
            return predict


