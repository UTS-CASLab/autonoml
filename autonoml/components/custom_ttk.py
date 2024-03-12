# -*- coding: utf-8 -*-
"""
Custom components designed/adapted by Thanh Tung Khuat.

Features Online Support Vector Regression (OSVR).
Adapted from MATLAB code available at: http://onlinesvr.altervista.org/

Created on Fri Nov 24 22:47:16 2023

@author: Thanh T. Khuat, David J. Kedziora
"""

import sys
import numpy as np

def sign(x):
    """ Returns sign. Numpys sign function returns 0 instead of 1 for zero values. """
    if x >= 0:
        return 1
    else:
        return -1

class OnlineSVR:
    """
    C is the regularization parameter, essentially defining the limit on how close the learner must adhere to the dataset (smoothness).
    Epsilon is the acceptable error, and defines the width of what is sometimes called the "SVR tube".
    The kernel parameter (gamma) is the scaling factor for comparing feature distance (this implementation uses a Radial Basis Function). 
    """

    def __init__(self, n_features, C, eps, gamma, bias = 0, debug = False):
        # Configurable Parameters
        self.n_features = n_features
        self.C = C
        self.eps = eps
        self.gamma = gamma
        self.bias = bias
        self.debug = debug
        
        # Algorithm initialization
        self.n_samples_trained = 0
        self.weights = np.array([])
        
        # Samples X (features) and Y (truths)
        self.X = list()
        self.Y = list()
        # Working sets, contains indices pertaining to X and Y
        self.support_set_indices = list()
        self.error_set_indices = list()
        self.remainder_set_indices = list()
        self.R = np.matrix([])

    def find_min_variation(self, H, beta, gamma, i):
        """ Finds the variations of each sample to the new set.
        Lc1: distance of the new sample to the SupportSet
        Lc2: distance of the new sample to the ErrorSet
        Ls(i): distance of the support samples to the ErrorSet/RemainingSet
        Le(i): distance of the error samples to the SupportSet
        Lr(i): distance of the remaining samples to the SupportSet
        """
        # Find direction q of the new sample
        q = -sign(H[i])
        # Compute variations
        Lc1 = self.find_var_lc1(H, gamma, q, i)
        q = sign(Lc1)
        Lc2 = self.find_var_lc2(H, q, i)
        Ls = self.find_var_ls(H, beta, q)
        Le = self.find_var_le(H, gamma, q)
        Lr = self.find_var_lr(H, gamma, q)
        # Check for duplicate minimum values, grab one with max gamma/beta, set others to inf
        # Support set
        if Ls.size > 1:
            minS = np.abs(Ls).min()
            results = np.array([k for k,val in enumerate(Ls) if np.abs(val)==minS])
            if len(results) > 1:
                betaIndex = beta[results+1].argmax()
                Ls[results] = q*np.inf
                Ls[results[betaIndex]] = q*minS
        # Error set
        if Le.size > 1:
            minE = np.abs(Le).min()
            results = np.array([k for k,val in enumerate(Le) if np.abs(val)==minE])
            if len(results) > 1:
                errorGamma = gamma[self.error_set_indices]
                gammaIndex = errorGamma[results].argmax()
                Le[results] = q*np.inf
                Le[results[gammaIndex]] = q*minE
        # Remainder Set
        if Lr.size > 1:
            minR = np.abs(Lr).min()
            results = np.array([k for k,val in enumerate(Lr) if np.abs(val)==minR])
            if len(results) > 1:
                remGamma = gamma[self.remainder_set_indices]
                gammaIndex = remGamma[results].argmax()
                Lr[results] = q*np.inf
                Lr[results[gammaIndex]] = q*minR
        
        # Find minimum absolute variation of all, retain signs. Flag determines set-switching cases.
        minLsIndex = np.abs(Ls).argmin()
        minLeIndex = np.abs(Le).argmin()
        minLrIndex = np.abs(Lr).argmin()
        minIndices = [None, None, minLsIndex, minLeIndex, minLrIndex]
        Ls = Ls.flatten()
        Le = Le.flatten()
        Lr = Lr.flatten()
        minValues = np.array([Lc1, Lc2, Ls[minLsIndex], Le[minLeIndex], Lr[minLrIndex]])

        if np.abs(minValues).min() == np.inf:
            # print('No weights to modify! Something is wrong.')
            print('n_features = ', self.n_features, ', C = ', self.C, ', eps = ', self.eps, ', gamma = ', self.gamma)
            sys.exit()
        flag = np.abs(minValues).argmin()
        if self.debug:
            print('MinValues',minValues)
        return minValues[flag], flag, minIndices[flag]

    def find_var_lc1(self, H, gamma, q, i):
        # weird hacks below
        Lc1 = np.nan
        if gamma.size < 2:
            g = gamma[0]
        else:
            g = gamma.item(i)
        # weird hacks above

        if  g <= 0:
            Lc1 = np.array(q*np.inf)
        elif H[i] > self.eps and -self.C < self.weights[i] and self.weights[i] <= 0:
            Lc1 = (-H[i] + self.eps) / g
        elif H[i] < -self.eps and 0 <= self.weights[i] and self.weights[i] <= self.C:
            Lc1 = (-H[i] - self.eps) / g
        #else:
            #print('Something is weird.')
            #print('i',i)
            #print('q',q)
            #print('gamma',gamma)
            #print('g',g)
            #print('H[i]',H[i])
            #print('weights[i]',self.weights[i])
        
        if np.isnan(Lc1):
            Lc1 = np.array(q*np.inf)
        return Lc1.item()

    def find_var_lc2(self, H, q, i):
        if len(self.support_set_indices) > 0:
            if q > 0:
                Lc2 = -self.weights[i] + self.C
            else:
                Lc2 = -self.weights[i] - self.C
        else:
            Lc2 = np.array(q*np.inf)
        if np.isnan(Lc2):
            Lc2 = np.array(q*np.inf)
        return Lc2

    def find_var_ls(self, H, beta, q):
        if len(self.support_set_indices) > 0 and len(beta) > 0:
            Ls = np.zeros([len(self.support_set_indices),1])
            supportWeights = self.weights[self.support_set_indices]
            supportH = H[self.support_set_indices]
            for k in range(len(self.support_set_indices)):
                if q*beta[k+1] == 0:
                    Ls[k] = q*np.inf
                elif q*beta[k+1] > 0:
                    if supportH[k] > 0:
                        if supportWeights[k] < -self.C:
                            Ls[k] = (-supportWeights[k] - self.C) / beta[k+1]
                        elif supportWeights[k] <= 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                    else:
                        if supportWeights[k] < 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        elif supportWeights[k] <= self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                else:
                    if supportH[k] > 0:
                        if supportWeights[k] > 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        elif supportWeights[k] >= -self.C:
                            Ls[k] = (-supportWeights[k] - self.C) / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                    else:
                        if supportWeights[k] > self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k+1]
                        elif supportWeights[k] >= self.C:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
        else:
            Ls = np.array([q*np.inf])

        # Correct for NaN
        Ls[np.isnan(Ls)] = q*np.inf
        if Ls.size > 1:
            Ls.shape = (len(Ls),1)
            # Check for broken signs
            for val in Ls:
                if sign(val) == -sign(q) and val != 0:
                    # print('Sign mismatch error in Ls! Exiting.')
                    sys.exit()
        # print('find_var_ls',Ls)
        return Ls
        
    def find_var_le(self, H, gamma, q):
        if len(self.error_set_indices) > 0:
            Le = np.zeros([len(self.error_set_indices),1])
            errorGamma = gamma[self.error_set_indices]
            errorWeights = self.weights[self.error_set_indices]
            errorH = H[self.error_set_indices]
            for k in range(len(self.error_set_indices)):
                if q*errorGamma[k] == 0:
                    Le[k] = q*np.inf
                elif q*errorGamma[k] > 0:
                    if errorWeights[k] > 0:
                        if errorH[k] < -self.eps:
                            Le[k] = (-errorH[k] - self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                    else:
                        if errorH[k] < self.eps:
                            Le[k] = (-errorH[k] + self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                else:
                    if errorWeights[k] > 0:
                        if errorH[k] > -self.eps:
                            Le[k] = (-errorH[k] - self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                    else:
                        if errorH[k] > self.eps:
                            Le[k] = (-errorH[k] + self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
        else:
            Le = np.array([q*np.inf])

        # Correct for NaN
        Le[np.isnan(Le)] = q*np.inf
        if Le.size > 1:
            Le.shape = (len(Le),1)
            # Check for broken signs
            for val in Le:
                if sign(val) == -sign(q) and val != 0:
                    # print('Sign mismatch error in Le! Exiting.')
                    sys.exit()
        # print('find_var_le',Le)
        return Le

    def find_var_lr(self, H, gamma, q):
        if len(self.remainder_set_indices) > 0:
            Lr = np.zeros([len(self.remainder_set_indices),1])
            remGamma = gamma[self.remainder_set_indices]
            remH = H[self.remainder_set_indices]
            for k in range(len(self.remainder_set_indices)):
                if q*remGamma[k] == 0:
                    Lr[k] = q*np.inf
                elif q*remGamma[k] > 0:
                    if remH[k] < -self.eps:
                        Lr[k] = (-remH[k] - self.eps) / remGamma[k]
                    elif remH[k] < self.eps:
                        Lr[k] = (-remH[k] + self.eps) / remGamma[k]
                    else:
                        Lr[k] = q*np.inf
                else:
                    if remH[k] > self.eps:
                        Lr[k] = (-remH[k] + self.eps) / remGamma[k]
                    elif remH[k] > -self.eps:
                        Lr[k] = (-remH[k] - self.eps) / remGamma[k]
                    else:
                        Lr[k] = q*np.inf
        else:
            Lr = np.array([q*np.inf])

        # Correct for NaN
        Lr[np.isnan(Lr)] = q*np.inf
        if Lr.size > 1:
            Lr.shape = (len(Lr),1)
            # Check for broken signs
            for val in Lr:
                if sign(val) == -sign(q) and val != 0:
                    # print('Sign mismatch error in Lr! Exiting.')
                    sys.exit()
        # print('find_var_lr',Lr)
        return Lr

    def compute_kernel_output(self, set1, set2):
        """Compute kernel output. Uses a radial basis function kernel."""
        X1 = np.matrix(set1)
        X2 = np.matrix(set2).T
        # Euclidean distance calculation done properly
        [S,R] = X1.shape
        [R2,Q] = X2.shape
        X = np.zeros([S,Q])
        if Q < S:
            copies = np.zeros(S,dtype=int)
            for q in range(Q):
                if self.debug:
                    print('X1',X1)
                    print('X2copies',X2.T[q+copies,:])
                    print('power',np.power(X1-X2.T[q+copies,:],2))
                xsum = np.sum(np.power(X1-X2.T[q+copies,:],2),axis=1)
                xsum.shape = (xsum.size,)
                X[:,q] = xsum
        else:
            copies = np.zeros(Q,dtype=int)
            for i in range(S):
                X[i,:] = np.sum(np.power(X1.T[:,i+copies]-X2,2),axis=0)
        X = np.sqrt(X)
        y = np.matrix(np.exp(-self.gamma*X**2))
        if self.debug:
            print('distance',X)
            print('kernelOutput',y)
        return y
    
    def predict(self, X):
        cur_X = np.array(self.X)
        new_X = np.array(X)
        weights = np.array(self.weights)
        weights.shape = (weights.size,1)
        if self.n_samples_trained > 0:
            y = self.compute_kernel_output(cur_X, new_X)
            return (weights.T @ y).T + self.bias
        else:
            return np.zeros_like(new_X) + self.bias

    def compute_margin(self, new_sample_X, new_sample_Y):
        fx = self.predict(new_sample_X)
        new_sample_Y = np.array(new_sample_Y)
        new_sample_Y.shape = (new_sample_Y.size, 1)
        if self.debug:
            print('fx',fx)
            print('new_sample_Y',new_sample_Y)
            print('hx',fx-new_sample_Y)
        return fx - new_sample_Y

    def compute_beta_gamma(self,i):
        """Returns beta and gamma arrays."""
        # Compute beta vector
        X = np.array(self.X)
        Qsi = self.compute_q(X[self.support_set_indices,:], X[i,:])
        if len(self.support_set_indices) == 0 or self.R.size == 0:
            beta = np.array([])
        else:
            beta = -self.R @ np.append(np.matrix([1]),Qsi,axis=0)
        # Compute gamma vector
        Qxi = self.compute_q(X, X[i,:])
        Qxs = self.compute_q(X, X[self.support_set_indices,:])
        if len(self.support_set_indices) == 0 or Qxi.size == 0 or Qxs.size == 0 or beta.size == 0:
            gamma = np.array(np.ones_like(Qxi))
        else:
            gamma = Qxi + np.append(np.ones([self.n_samples_trained,1]), Qxs, 1) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0
        if self.debug:
            print('R',self.R)
            print('beta',beta)
            print('gamma',gamma)
        return beta, gamma

    def compute_q(self, set1, set2):
        set1 = np.matrix(set1)
        set2 = np.matrix(set2)
        Q = np.matrix(np.zeros([set1.shape[0],set2.shape[0]]))
        for i in range(set1.shape[0]):
            for j in range(set2.shape[0]):
                Q[i,j] = self.compute_kernel_output(set1[i,:],set2[j,:])
        return np.matrix(Q)
        
    def adjust_sets(self, H, beta, gamma, i, flag, minIndex):
        # print('Entered adjustSet logic with flag {0} and minIndex {1}.'.format(flag,minIndex))
        if flag not in range(5):
            print('Received unexpected flag {0}, exiting.'.format(flag))
            sys.exit()
        # add new sample to Support set
        if flag == 0:
            # print('Adding new sample {0} to support set.'.format(i))
            H[i] = np.sign(H[i])*self.eps
            self.support_set_indices.append(i)
            self.R = self.add_sample_to_r(i,'SupportSet',beta,gamma)
            return H,True
        # add new sample to Error set
        elif flag == 1: 
            # print('Adding new sample {0} to error set.'.format(i))
            self.weights[i] = np.sign(self.weights[i])*self.C
            self.error_set_indices.append(i)
            return H,True
        # move sample from Support set to Error or Remainder set
        elif flag == 2: 
            index = self.support_set_indices[minIndex]
            weightsValue = self.weights[index]
            if np.abs(weightsValue) < np.abs(self.C - abs(weightsValue)):
                self.weights[index] = 0
                weightsValue = 0
            else:
                self.weights[index] = np.sign(weightsValue)*self.C
                weightsValue = self.weights[index]
            # Move from support to remainder set
            if weightsValue == 0:
                # print('Moving sample {0} from support to remainder set.'.format(index))
                self.remainder_set_indices.append(index)
                self.R = self.remove_sample_from_r(minIndex)
                self.support_set_indices.pop(minIndex)
            # move from support to error set
            elif np.abs(weightsValue) == self.C:
                # print('Moving sample {0} from support to error set.'.format(index))
                self.error_set_indices.append(index)
                self.R = self.remove_sample_from_r(minIndex)
                self.support_set_indices.pop(minIndex)
            else:
                # print('Issue with set swapping, flag 2.','weightsValue:',weightsValue)
                sys.exit()
        # move sample from Error set to Support set
        elif flag == 3: 
            index = self.error_set_indices[minIndex]
            # print('Moving sample {0} from error to support set.'.format(index))
            H[index] = np.sign(H[index])*self.eps
            self.support_set_indices.append(index)
            self.error_set_indices.pop(minIndex)
            self.R = self.add_sample_to_r(index, 'ErrorSet', beta, gamma)
        # move sample from Remainder set to Support set
        elif flag == 4: 
            index = self.remainder_set_indices[minIndex]
            # print('Moving sample {0} from remainder to support set.'.format(index))
            H[index] = np.sign(H[index])*self.eps
            self.support_set_indices.append(index)
            self.remainder_set_indices.pop(minIndex)
            self.R = self.add_sample_to_r(index, 'RemainingSet', beta, gamma)
        return H, False

    def add_sample_to_r(self, sampleIndex, sampleOldSet, beta, gamma):
        # print('Adding sample {0} to R matrix.'.format(sampleIndex))
        X = np.array(self.X)
        sampleX = X[sampleIndex,:]
        sampleX.shape = (sampleX.size//self.n_features, self.n_features)
        # Add first element
        if self.R.shape[0] <= 1:
            Rnew = np.ones([2,2])
            Rnew[0,0] = -self.compute_kernel_output(sampleX,sampleX)
            Rnew[1,1] = 0
        # Other elements
        else:
            # recompute beta/gamma if from error/remaining set
            if sampleOldSet == 'ErrorSet' or sampleOldSet == 'RemainingSet':
                # beta, gamma = self.compute_beta_gamma(sampleIndex)
                Qii = self.compute_kernel_output(sampleX, sampleX)
                Qsi = self.compute_kernel_output(X[self.support_set_indices[0:-1],:], sampleX)
                beta = -self.R @ np.append(np.matrix([1]),Qsi,axis=0)
                beta[np.isnan(beta)] = 0
                beta.shape = (len(beta),1)
                gamma[sampleIndex] = Qii + np.append(1,Qsi.T)@beta
                gamma[np.isnan(gamma)] = 0
                gamma.shape = (len(gamma),1)
            # add a column and row of zeros onto right/bottom of R
            r,c = self.R.shape
            Rnew = np.append(self.R, np.zeros([r,1]), axis=1)
            Rnew = np.append(Rnew, np.zeros([1,c+1]), axis=0)
            # update R
            if gamma[sampleIndex] != 0:
                # Numpy so wonky! SO WONKY.
                beta1 = np.append(beta, [[1]], axis=0)
                Rnew = Rnew + 1/gamma[sampleIndex].item()*beta1@beta1.T
            if np.any(np.isnan(Rnew)):
                # print('R has become inconsistent. Training failed at sampleIndex {0}'.format(sampleIndex))
                sys.exit()
        return Rnew

    def remove_sample_from_r(self, sampleIndex):
        # print('Removing sample {0} from R matrix.'.format(sampleIndex))
        sampleIndex += 1
        I = list(range(sampleIndex))
        I.extend(range(sampleIndex+1,self.R.shape[0]))
        I = np.array(I)
        I.shape = (1,I.size)
        if self.debug:
            print('I',I)
            print('RII',self.R[I.T,I])
        # Adjust R
        if self.R[sampleIndex,sampleIndex] != 0:
            Rnew = self.R[I.T,I] - (self.R[I.T,sampleIndex]*self.R[sampleIndex,I]) / self.R[sampleIndex,sampleIndex].item()
        else:
            Rnew = np.copy(self.R[I.T,I])
        # Check for bad things
        if np.any(np.isnan(Rnew)):
            # print('R has become inconsistent. Training failed removing sampleIndex {0}'.format(sampleIndex))
            sys.exit()
        if Rnew.size == 1:
            # print('Time to annhilate R? R:',Rnew)
            Rnew = np.matrix([])
        return Rnew

    def learn(self, new_X, new_Y):
        self.n_samples_trained += 1
        self.X.append(new_X)
        self.Y.append(new_Y)
        self.weights = np.append(self.weights,0)
        i = self.n_samples_trained - 1 # stupid off-by-one errors
        H = self.compute_margin(self.X, self.Y)

        # correctly classified sample, skip the rest of the algorithm!
        if (abs(H[i]) <= self.eps):
            # print('Adding new sample {0} to remainder set, within eps.'.format(i))
            if self.debug:
                print('weights',self.weights)
            self.remainder_set_indices.append(i)
            return

        is_new_sample_added = False
        iterations = 0
        while not is_new_sample_added:
            # Ensure we're not looping infinitely
            iterations += 1
            if iterations > self.n_samples_trained*100:
                # print('Warning: we appear to be in an infinite loop.')
                sys.exit()
                iterations = 0
            # Compute beta/gamma for constraint optimization
            beta, gamma = self.compute_beta_gamma(i)
            # Find minimum variation and determine how we should shift samples between sets
            deltaC, flag, minIndex = self.find_min_variation(H, beta, gamma, i)
            # Update weights and bias based on variation
            if len(self.support_set_indices) > 0 and len(beta)>0:
                self.weights[i] += deltaC
                delta = beta*deltaC
                self.bias += delta.item(0)
                # numpy is wonky...
                weight_delta = np.array(delta[1:])
                weight_delta.shape = (len(weight_delta),)
                self.weights[self.support_set_indices] += weight_delta
                H += gamma*deltaC
            else:
                self.bias += deltaC
                H += deltaC
            # Adjust sets, moving samples between them according to flag
            H, is_new_sample_added = self.adjust_sets(H, beta, gamma, i, flag, minIndex)
        
        if self.debug:
            print('weights',self.weights)



#%% The MLComponents.

from ..hyperparameter import HPFloat
from ..component import MLOnlineLearner, MLRegressor
from ..data import DataFormatX, DataFormatY

from typing import List

class OnlineSupportVectorRegressor(MLRegressor, MLOnlineLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Warning: Components are typically initialised without knowing data ahead of time.
        #          Currently, number of features is unknown until a surrounding pipeline is constructed.
        #          Therefore, this component should not be used outside of a pipeline without care.
        # TODO: Reconsider whether MLComponent should require an in_keys_features argument.
        self.model = None
        self.name += "_CustomTTK_OnlineSVR"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D
        self.format_y = DataFormatY.NUMPY_ARRAY_2D
        self.is_setup_complete = False

    @staticmethod
    def new_hpars():
        hpars = dict()
        info = ("Regularisation parameter, defining the limit on how close "
                "the learner must adhere to the dataset (smoothness).")
        hpars["C"] = HPFloat(in_default = 1.0, in_min = 0.1, in_max = 10.0,
                             is_log_scale = True, in_info = info)
        info = ("The acceptable error, defining the width of what is sometimes "
                "called the 'SVR tube'.")
        hpars["epsilon"] = HPFloat(in_default = 0.0, in_min = 0.0, in_max = 1.0,
                                   in_info = info)
        info = ("The kernel parameter, which is the scaling factor for comparing feature distance. "
                "This implementation uses a Radial Basis Function.")
        hpars["gamma"] = HPFloat(in_default = 0.01, in_min = 0.0001, in_max = 1.0,
                                 is_log_scale = True, in_info = info)
        return hpars

    def learn(self, x, y):
        if not self.is_setup_complete:
            raise Exception(f"Delayed setup for {self.name} was not completed before methods were called.")
        for x_i, y_i in zip(x, y):
            self.model.learn(new_X = x_i, new_Y = y_i)

    def query(self, x):
        if not self.is_setup_complete:
            raise Exception(f"Delayed setup for {self.name} was not completed before methods were called.")
        responses = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            x_i = x[i, :]
            y_i = self.model.predict(X = x_i)
            responses[i, 0] = y_i
        # print(responses)
        return responses
    
    def set_keys_features(self, in_keys_features: List[str]):
        """
        Custom utility function overwrite.
        This is required for updating the model with the number of features to expect.
        """
        self.keys_features = in_keys_features
        self.run_delayed_setup()

    def run_delayed_setup(self):
        self.model = OnlineSVR(n_features = len(self.keys_features),
                               C = self.hpars["C"].val, 
                               eps = self.hpars["epsilon"].val, 
                               gamma = self.hpars["gamma"].val)
        self.is_setup_complete = True