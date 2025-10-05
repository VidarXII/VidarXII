# Neural Ordinary Differential Equations: From Theory to Implementation

**Author:** Nishant Padmanabhan  
**Date:** Today

---

## Abstract
Neural Ordinary Differential Equations (Neural ODEs) provide a framework for modeling continuous dynamics with neural networks. This report explores the logic, mathematical underpinnings, and implementation of Neural ODEs, particularly in the context of recognizing spiral-shaped differential equations.

---

## Introduction
Ordinary Differential Equations (ODEs) are equations that relate a function to its derivatives. In recent years, they have inspired the development of continuous-depth neural networks called Neural ODEs. These models offer an elegant alternative to traditional feedforward networks by treating hidden layers as a continuous transformation parameterized by an ODE solver.

---

## Background: From ODEs to Neural ODEs
Let **h(t)** be the hidden state at time *t*. A Neural ODE models the evolution of this state as:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t; \theta)$$

where *f* is a neural network and *θ* its parameters. Instead of stacking layers, the final state is obtained by integrating this differential equation:

$$\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t; \theta)\,dt$$

---

## Adjoint Sensitivity Method
To train a Neural ODE, we must differentiate through the ODE solver. Instead of backpropagating through all solver steps, the adjoint sensitivity method solves a backward ODE to compute gradients efficiently.

Let **L** be the loss. Define the adjoint state as:

$$\mathbf{a}(t) = \frac{\partial \mathcal{L}}{\partial \mathbf{h}(t)}$$

The dynamics of the adjoint state follow:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t; \theta)}{\partial \mathbf{h}}$$

Additional gradients (e.g., with respect to θ) can be computed as:

$$\frac{d\mathcal{L}}{d\theta} = - \int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t; \theta)}{\partial \theta} dt$$

---

## Code Snippets

### ODE Function Definition
```python
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t, y):
        return self.net(y)
```

### Model Integration and Training
```python
odeblock = torchdiffeq.odeint(ODEFunc(), y0, t)
# Training loop omitted for brevity
```

---

## Results

![Neural ODE predictions vs true spiral trajectories](spiral_results.png)

*Figure: Neural ODE predictions vs true spiral trajectories.*

---

## Conclusion
Neural ODEs provide a flexible and interpretable framework for modeling continuous-time dynamics. With the adjoint sensitivity method, these models become scalable and differentiable, opening up new possibilities in time-series modeling and physics-inspired learning.

---

## Appendix A: Proof of Adjoint Method
We define the augmented hidden state **z(t) = [h(t), θ, t]**.

$$\frac{d\mathcal{L}}{dt} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} f(\mathbf{h}(t), t; \theta)$$

and

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f}{\partial \mathbf{h}}(\mathbf{h}(t), t; \theta)$$

and finally,

$$\frac{d\mathcal{L}}{d\theta} = - \int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t; \theta)}{\partial \theta} dt$$

---

## Appendix B: Gradients with Respect to Parameters

$$\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t; \theta)}{\partial \theta} dt$$

---

## Appendix C: Proof of Instantaneous Change of Variables Formula

To prove:

$$\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\mathrm{tr}\left(\frac{\partial f}{\partial \mathbf{z}}(t)\right)$$

Through Taylor series expansion:

$$T_{\varepsilon}(\mathbf{z}(t)) = \mathbf{z}(t) + \varepsilon f(\mathbf{z}(t), t) + \mathcal{O}(\varepsilon^2)$$

and taking limits yields:

$$\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\mathrm{tr}\left(\frac{\partial f(\mathbf{z}(t), t)}{\partial \mathbf{z}}\right)$$

---

## Appendix D: Continuous Backpropagation and Adjoint Equation

Given:

$$\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta)$$

Define:

$$\mathbf{a}(t) = \frac{d\mathcal{L}}{d\mathbf{z}(t)}$$

Then:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t) \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)}$$

and integrating backward in time:

$$\mathbf{a}(t_0) = \mathbf{a}(t_N) - \int_{t_N}^{t_0} \mathbf{a}(t) \cdot \frac{\partial f(\mathbf{z}(t), t, \theta)}{\partial \mathbf{z}(t)} dt$$

If **L** depends on intermediate time points, the adjoint step is repeated backward across all intervals, accumulating gradients.

---

## References
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural ordinary differential equations*. NeurIPS.
- Rackauckas, C., & Nie, Q. (2017). *DifferentialEquations.jl – A performant and feature-rich ecosystem for solving differential equations in Julia*. *Journal of Open Research Software*.
- [torchdiffeq GitHub Repository](https://github.com/rtqichen/torchdiffeq)
