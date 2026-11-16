import numpy as np

class SchreiFilterMethod:

    def __init__(self,x, l, k, epsilon):
        self.x = np.array(x)
        self.l = l
        self.k = k
        self.epsilon = epsilon


    def create_window (self):

        windows = []
        n = len(self.x)

        # Center index i must leave room for k past points and l future points
        for i in range(self.k, n - self.l):
            window = self.x[i - self.k : i + self.l + 1]
            windows.append(window)

        return np.array(windows)


    def clean_algo (self, x):
        """
        Arguments:
        Input : Time Series Data, x
        """
        pass


        
if __name__ == "__main__":

    x = np.arange(0, 10, 1)  
    l = k = 1
    epsilon = 0.1
    SFM = SchreiFilterMethod(x, l,k,epsilon)
    
    window = SFM.create_window()
    print(f"{window.ndim}")
    print(f"{window}")


