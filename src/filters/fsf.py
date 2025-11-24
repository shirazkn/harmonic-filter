"""
A class for the Fourier Series Filter (FSF)
"""
import numpy as np
import torch as t
from scipy.stats import multivariate_normal

from scipy.fft import fftshift, ifftshift, fftn, ifftn

def lie_derivative_matrix(ind, k: list[int]=None, 
                          bandwidth: int=None, 
                          domain_length=2*t.pi,):
    i = 1j
    pol = t.as_tensor(t.pi/domain_length)
    if ind in [1, 2]:
        k1 = k[0]
        k2 = k[1]
        if ind == 1:
            upper = i * pol * k1 - pol * k2
            lower = i * pol * k1 + pol * k2
        elif ind == 2:
            upper = i * pol * k2 + pol * k1
            lower = i * pol * k2 - pol * k1
        
        return t.diag(upper.repeat(2*bandwidth), 1) \
            + t.diag(lower.repeat(2*bandwidth), -1)
    
    elif ind == 3:
        return t.diag(t.arange(-bandwidth, bandwidth+1))

class FourierSeriesFilter:
    def __init__(
        self,
        prior: np.ndarray,
        prior_cov: np.ndarray,
        grid_size: tuple[int, int, int],
        grid_samples: np.ndarray):
        """
        :param prior: Prior mean.
        :param prior_cov: Prior covariance.
        """
        self.pose_grid = grid_samples
        self.prior = multivariate_normal(prior, prior_cov).pdf(self.pose_grid)
        self.prior /= np.sum(self.prior)
        self.bandwidths = grid_size

    def prediction(self, step, diffusion_coefficient, dt):
        """
        Prediction step
        :param motion_model: motion model for prediction step
        :return unnormalized belief distribution
        """ 
        #So this gives the vector of fourier coefficients as seen in equation 16
        import pdb; pdb.set_trace()
        f_hat = fftshift(
            fftn(
                ifftshift(self.prior),
                norm='forward'
                )
            )
        
        return self.prior

    def update(self,
               map_mask: np.ndarray,
               landmarks: np.ndarray,
               observations: np.ndarray,
               observations_cov: np.ndarray) -> np.ndarray:
        """
        Update step
        :param map_mask: Binary mask indicating traversable area
        :param landmarks: location of each UWB landmark in the map (n, 3)
        :param observations: range measurements of dimension (n,)
        :param observations_cov: variance of each measurement of dimension (n,)
        :return Mean of the particles
        """
        # observations_std = np.sqrt(observations_cov)
        # ### Not independent measurements ###
        # # weight = 1 / len(observations_std)
        # # measurement_likelihood = 1e-300
        # # for i, landmark in enumerate(landmarks):
        # #     dist = np.linalg.norm(landmark - self.grid_samples[:, :2], axis=1)
        # #     measurement_likelihood += norm(dist, observations_std[i]).pdf(observations[i]) * weight
        # ### Independence between measurements ###
        # measurement_likelihood = 0.
        # for i, landmark in enumerate(landmarks):
        #     dist = np.linalg.norm(landmark - self.grid_samples[:, :2], axis=1)
        #     prob = norm(dist, observations_std[i]).pdf(observations[i]) + 1e-8
        #     measurement_likelihood += np.log(prob)

        # measurement_likelihood -= logsumexp(measurement_likelihood)
        # if measurement_likelihood is not None:
        #     # Combine the prior belief and the measurement likelihood to get the posterior belief
        #     p_belief = self.prior * np.exp(measurement_likelihood)
        #     # Normalizing the posterior belief
        #     if np.sum(p_belief) != 0.:
        #         self.prior = p_belief / np.sum(p_belief)

        # # Compute mean of histogram filter
        # mean = self._compute_mean()
        # return mean
        return

    def neg_log_likelihood(self, pose) -> np.ndarray:
        """
        Evaluate posterior distribution of histogram filter
        :param pose: Pose at which to interpolate the SE2 Fourier transform
        :return ll: Probability of distribution determined by fourier coefficients (moments) at given pose
        """
        # Grid samples are between [0, 2pi] so we need to transform the pose to that range
        # wrapped_pose = pose.copy()
        # wrapped_pose[2] = wrapped_pose[2] % (2 * np.pi)
        # idx = np.argmin(np.linalg.norm(self.grid_samples 
        #                                - wrapped_pose, axis=-1))
        # # Divide by cube's volume to obtain pdf
        # ll = np.log((self.prior[idx] / self.volume) + 1e-8)
        return 0.0
    
    def compute_mode(self) -> np.ndarray:
        """
        Compute mode (maximizer of pdf)
        :return mode
        """
        return self.pose_grid[0]

    def compute_mean(self) -> np.ndarray:
        """
        Compute the projected Euclidean mean
        :return mean
        """
        # We do not want to do the following, but rather use a projected mean
        # prod = self.grid_samples * rearrange(self.prior, 'n -> n 1')
        # mean = np.sum(prod, axis=0)
        return self.pose_grid[0]