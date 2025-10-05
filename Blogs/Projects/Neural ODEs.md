# Neural Ordinary Differential Equations: Theory and Implementation on Spiral Datasets

**Author:** Nishant Padmanabhan  
**Date:** June 2025

---

## Abstract
Neural Ordinary Differential Equations (Neural ODEs) provide a framework for modeling continuous dynamics with neural networks. This report explores the logic, mathematical underpinnings, and implementation of Neural ODEs in the context of recognizing and predicting spiral-shaped differential equations.

---

## Introduction
Ordinary Differential Equations (ODEs) are equations that relate a function to its derivatives. In recent years, they have inspired the development of continuous-depth neural networks called Neural ODEs. These models offer an elegant alternative to traditional feedforward networks by treating hidden layers as a continuous transformation parameterized by an ODE solver.

Consider a Residual network (ResNet) with skip connections. The mathematical equation behind such a model would look like:

```math
\mathbf{h}_2 = x + f(\mathbf{h}_1)
```

where **x** is the input and **f** is a nonlinear function such as `tanh(h₁·w + bᵢ)`. Interestingly, this structure closely resembles a differential equation.

---

## Background: From ODEs to Neural ODEs
To define any differential equation, one needs two things: the initial state and the bounded gradient at each point. The logic behind using Neural ODEs is that it assumes **h(t₀)** as the input and **h(t₁)** as the output for two consecutive layers.

```math
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t; \theta)
```

The final state is then computed by integrating this differential equation:

```math
\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t; \theta)\,dt
```

---

## Adjoint Sensitivity Method
To train a Neural ODE, we must differentiate through the ODE solver. Instead of backpropagating through all solver steps, the adjoint sensitivity method solves a backward ODE to compute gradients efficiently.

### Proof of Instantaneous Change of Variables Formula
We prove:

```math
\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\mathrm{tr} \left( \frac{\partial f}{\partial \mathbf{z}}(t) \right)
```

Using Taylor expansion and limit arguments, the proof concludes that:

```math
\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\mathrm{tr}\left(\frac{\partial f(\mathbf{z}(t), t)}{\partial \mathbf{z}}\right)
```

### Continuous Backpropagation and Adjoint Equation
Given:

```math
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)
```

Define the adjoint:

```math
\mathbf{a}(t) = \frac{d\mathcal{L}}{d\mathbf{z}(t)}
```

Then:

```math
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}
```

and integrating backward in time:

```math
\mathbf{a}(t_0) = \mathbf{a}(t_N) - \int_{t_N}^{t_0} \mathbf{a}(t) \cdot \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} dt
```

If **L** depends on intermediate time points, the adjoint step is repeated backward across all intervals, accumulating gradients.

---

## My Project
The experiment uses the `torchdiffeq` package to train a Neural ODE model on a spiral dataset defined as:

```math
x = \cos(t), \quad y = \sin(t)
```

The goal is to test generalization on altered spirals by varying spiral radius and damping rate.

Hyperparameters used include:  
- Hidden layer dimension: 256  
- ODE Solver: `rk4` (chosen for efficiency)  
- Relative tolerance: 1e-9  
- Absolute tolerance: 1e-10  
- Learning rate: 1e-3  
- Early stopping based on validation loss

---

## Code Snippets

### ODE Function Definition
```python
class GeneralODEFunc(nn.Module):
    def __init__(self, input_dim, param_dim=0, hidden_dim=128, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.param_dim = param_dim
        layers = [nn.Linear(input_dim + param_dim + 1, hidden_dim), nn.Tanh(), nn.Dropout(0.1)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y_with_params):
        y, params = y_with_params[:, :self.input_dim], y_with_params[:, self.input_dim:]
        t_expanded = t.expand(y.shape[0], 1)
        inp = torch.cat([y, params, t_expanded], dim=1)
        return self.net(inp)
```

### Parameter Integration Helper
```python
def integrate_with_params(func, y0, params, t, input_dim, **kwargs):
    if y0.dim() == 1: y0 = y0.unsqueeze(0)
    if params.dim() == 1: params = params.unsqueeze(0)
    if y0.shape[0] > 1 and params.shape[0] == 1:
        params = params.expand(y0.shape[0], -1)
    y0_aug = torch.cat([y0, params], dim=1)
    pred_y_aug = odeint(func, y0_aug, t, **kwargs)
    return pred_y_aug[:, :, :input_dim]
```

### Model Integration and Training
```python
pred_y = integrate_with_params(self.func, y0_train, params, t_train, input_dim, ...)
# Training loop omitted for brevity
```

---

## Results
During training, the model learned to approximate spiral trajectories accurately. However, when tested on unseen spirals, performance decreased—suggesting overfitting and poor generalization.

![Training data]("Assets/Training_data.png")  
*Neural ODE trajectory vs actual data during training and validation.*

![Testing data]("Assets/Testing_data.png")  
*Neural ODE predictions vs true spiral trajectories during testing.*

---

## Conclusions and Future Enhancements
The Neural ODE framework captures continuous dynamics effectively. However, in this experiment, overfitting limited generalization.

### Future Suggestions
1. **Data-Centric Enhancements:** Train on a diverse range of spirals with randomized parameters *(a, b)* to encourage generalization.  
2. **Physics-Informed Priors:** Incorporate known dynamics (e.g., spiral ODE form) as priors, allowing the model to learn physical relationships instead of memorizing data.

---

## References
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural ordinary differential equations*. NeurIPS.  
- Rackauckas, C., & Nie, Q. (2017). *DifferentialEquations.jl – A performant and feature-rich ecosystem for solving differential equations in Julia.*  
- [torchdiffeq GitHub Repository](https://github.com/rtqichen/torchdiffeq)  
- Sinai, J. (2019). *Understanding Neural ODEs.* [jontysinai.github.io/posts](https://jontysinai.github.io/posts)  
- Gibson, K. (2018). *Neural networks as Ordinary Differential Equations.* [rkevingibson.github.io](https://rkevingibson.github.io/)  
- Hu, P. (2018). *A note on the adjoint method for neural ordinary differential equations.* [arxiv.org/2402.15141v1](https://arxiv.org/2402.15141v1)  
- NASA (1992). *Description and Use of LSODE, the Livermore Solver for Ordinary Differential Equations.*  
- Nair, A., Kadam, S. D., Menon, R., & Shende, P. C. (2024). *Neural Differential Equations: A Comprehensive Review and Applications.* *Advances in Nonlinear Variational Inequalities.*
