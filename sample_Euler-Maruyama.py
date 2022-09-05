import torch
import tqdm
from scoremodel import ScoreNet
import functools
import numpy as np
#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  1000#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

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
    sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

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
    plt.savefig(f'./Euler_Maruyama_sampler_numsteps_{num_steps}.png')
    plt.show()