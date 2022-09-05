import torch
import tqdm
from scoremodel import ScoreNet
import functools
import numpy as np
#@title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x


if __name__ == "__main__":
  #@title Sampling (double click to expand or collapse)

    from torchvision.utils import make_grid
    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

    def marginal_prob_std(t, sigma):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
        任意t时刻加噪后 高斯分布的标准差
        Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  

        Returns:
        The standard deviation.
        """    
        t = torch.tensor(t, device=device)
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

    def diffusion_coeff(t, sigma):
        """Compute the diffusion coefficient of our SDE.
        任意t时刻加噪后的扩散系数, 本例子没有漂移系数
        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

        Returns:
        The vector of diffusion coefficients.
        """
        return torch.tensor(sigma**t, device=device)
      
    sigma =  25.0#@param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma) # marginal_prob_std_fn(1) equals margin_prob_std(1, 25)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    ## Load the pre-trained checkpoint from disk.
    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
    ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    sample_batch_size = 64 #@param {'type':'integer'}
    sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Generate samples using the specified sampler.
    samples = sampler(score_model, 
                      marginal_prob_std_fn,
                      diffusion_coeff_fn, 
                      sample_batch_size, 
                      device=device)

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    # %matplotlib inline
    import matplotlib.pyplot as plt
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(f'./ODE_sampler.png')
    plt.show()