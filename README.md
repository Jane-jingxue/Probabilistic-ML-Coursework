# Advanced Probabilistic Machine Learning (SSY316)

**Chalmers University of Technology** | *Master's Coursework*

This branch contains the final project implementations for the course **Advanced Probabilistic Machine Learning (SSY316)**. The project covers three distinct pillars of modern probabilistic ML: Bayesian inference in graphical models, derivative-free optimization, and state-of-the-art generative modeling (Latent Diffusion).

### Part 1: Bayesian Skill Estimation with TrueSkill
**Keywords:** *Gaussian Process, Factor Graphs, Gibbs Sampling, Assumed Density Filtering (ADF), Message Passing*

Implementation of a Bayesian ranking system based on the TrueSkill algorithm to estimate player skills from match outcomes.

* **Model Formulation:** Modeled player skills as Gaussian random variables and match outcomes as truncated Gaussians within a Factor Graph framework.
* **Inference Algorithms:**
    * Implemented a **Gibbs Sampler** from scratch. Analysis showed a burn-in period of ~370 samples, with optimal posterior estimation achieved at ~2,000 samples.
    * Developed an online learning pipeline using **Assumed Density Filtering (ADF)** to process sequential match data from the 2018/2019 Serie A season.
    * Validated results by comparing Gibbs sampling distributions against a deterministic **Message Passing** algorithm using moment matching.
* **Extension: Asymmetric Prior & Performance Margins:**
    * Addressed the limitation of symmetric skill models by introducing an **asymmetric prior** to account for "first-mover" advantages (e.g., White in Chess, Home Team in Football).
    * Refined the observation model to incorporate a continuous performance margin proportional to match dominance ($y_k = \alpha m_k$).
    * **Results:** The extended model improved prediction accuracy by **+5.5%** on the Serie A dataset and **+0.8%** on a Kaggle Chess dataset.

### Part 2: Derivative-Free Optimization (QWOP)
**Keywords:** *Black-box Optimization, Simulated Annealing, Evolutionary Algorithms, MCMC*

Development of an optimization strategy for the physics-based game QWOP.
* **Objective:** Maximize the distance traveled by an avatar by optimizing a 40-dimensional vector controlling thigh and knee angles.
* **Methods Evaluated:**
    1. **Markov Chain Monte Carlo (MCMC)**: (Best distance: 6.94m) 
    2. **Evolutionary Algorithm**: (Best distance: 7.52m)
    3. **Simulated Annealing**: Implemented with a dynamic temperature schedule to escape local optima.
* **Results:** **Simulated Annealing** achieved the best performance with a total distance of **8.0988 meters**.

### Part 3: Deep Generative Models & Latent Diffusion
**Keywords:** *Variational Autoencoders (VAE), Score Matching, Stochastic Differential Equations (SDE), Latent Diffusion, Classifier-Free Guidance*

Implementation of modern generative pipelines using **PyTorch**.

* **Variational Autoencoder (VAE):**
    * Trained an MLP-based VAE on MNIST with a 2D latent space ($d_z=2$) for direct visualization of the data manifold.
    * Optimized the Evidence Lower Bound (ELBO) using the reparameterization trick.
* **Score-Based Diffusion:**
    * Modeled a Variance Preserving (VP) SDE to diffuse a 2D spiral distribution into noise.
    * Trained a score network to approximate $\nabla_x \log p_t(x)$ using the **Hyvärinen score matching objective**
    * Successfully generated clean spiral data from noise using the reverse-time SDE.
* **Latent Diffusion & Conditional Generation:**
    * Implemented diffusion in the VAE's latent space to reduce computational complexity.
    * **Classifier-Free Guidance (CFG):** To resolve overlapping clusters in the 2D latent space (e.g., digits "4" and "9"), implemented CFG with a guidance scale of $w=7.5$. This forced the model to disentangle classes and generate specific digits on demand.
