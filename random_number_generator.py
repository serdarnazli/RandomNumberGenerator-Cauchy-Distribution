import numpy as np



class RandomNumberGenerator:
    def __init__(self, x0, gamma, domain):
        """
        Input:
            x0: float, integer
            gamma: float, integer
            domain: 1D list or numpy.ndarray
        """
        seed = 1881
        np.random.seed(seed)
        self.x0 = x0
        self.gamma = gamma 
        self.domain = domain
    
    def pdf(self, x):
        """
        Input:
            x: int, float, 1D list, numpy.ndarray
        Output:
            pdf value of x: int, float, 1D list, numpy.ndarray
        """
        result = None

        # If x is integer or float;
        if isinstance(x, int) or isinstance(x,float):
            result = 1 / ( (np.pi * self.gamma)*(1+( (x-self.x0) / self.gamma)**2 ) )

        # If x is a numpy array, calculate every pdf value in numpy array
        elif isinstance(x, np.ndarray):
            # If x is not 1D raise ValueErr.
            if not (x == x.flatten()).all():
                raise ValueError(f"Given parameter must be 1D numpy.ndarray, Given dimension is: {x.shape}")

            result = np.array([self.pdf(number) for number in x])

        # If x is a list, calculate every pdf value in list.
        elif isinstance(x, list):
            # If x is not 1D, raise ValueErr.
            if not (isinstance(x[0],int) or isinstance(x[0],float)):
                raise ValueError("List must be 1D flatten list which consist of integers or floats.")

            result = [self.pdf(number) for number in x]

        else:
            # If x is not int,float,1dlist nor np.ndarray, then raise ValueErr.
            raise ValueError(f"Parameter x must be int, float, 1D list or numpy.ndaray. Given type is: {type(x)}")

        # Termination.
        return result
    
    def cdf(self, x):
        """
        Input:
            x: int, float, 1D list, numpy.ndarray
        Output:
            cdf value of x: int, float, 1D list, numpy.ndarray
        """
        result = None

        # If x is integer or float ;
        if isinstance(x, int) or isinstance(x, float) or isinstance(x,np.int64):
            result = float(  (1/np.pi) * np.arctan([ (x-self.x0) / self.gamma]) + 0.5  )
  
        # If x is np.ndarray then, calculate every cdf value in np.ndarray
        elif isinstance(x, np.ndarray):
            # If np.array is not 1D raise ValueErr.
            if not (x == x.flatten()).all():
                raise ValueError(f"Given parameter must be 1D numpy.ndarray, Given dimension is: {x.shape}")
            
            result  = np.array([self.cdf(number) for number in x])
        
        # If x is list; calculate every cdf value in list
        elif isinstance(x, list):
            # If list is not 1D, raise ValueErr.
            if not (isinstance(x[0], int) or isinstance(x[0], float)):
                raise ValueError("List must be 1D flatten list which consist of integers or floats. ")
            
            result = [self.cdf(number) for number in x]
        
        # If x is not int,float,1dlist nor np.ndarray, then raise ValueErr.
        else:
            raise ValueError(f"Parameter x must be int, float, 1D list or numpy.ndaray. Given type is: {type(x)}")

        # Termination
        return result
    
    def randint(self):
        """
        Output:
            random integer: int
        """
        result = None

        n = float(np.random.uniform(size=1))
        cdfs = [[index,self.cdf(index)] for index in self.domain]
        cdfs_smaller_n = [item for item in cdfs if item[1] < n]

        # If there is cdf value smaller than n, get the smallest cdf's index.
        if not cdfs_smaller_n:
            result = sorted(cdfs, key = lambda x:x[1])[0][0]
        
        # If there are cdf values smaller than n, get the biggest one's index which is smaller than n.
        else:
            result = sorted(cdfs_smaller_n, key = lambda x:x[1])[-1][0]
     
        return float(result)
    
    def __str__(self):
        """
            returns a string formatted with: "RandomNumberGenerator({self.x0},{self.gamma})"
        """
        return f"RandomNumberGenerator({self.x0},{self.gamma})"
    