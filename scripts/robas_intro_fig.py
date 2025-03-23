from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import expon
from bayesian_dro.Bayesian_DRO_continuous import data_generation
from mis_dro.dataset import data_generation_outliers, data_generation_gamma, contaminated_normal
from bayesian_dro.Bayesian_DRO_continuous import EPSILON_SET
from mis_dro.models import univariate_GaussianModel
from mis_dro.npl import Npl
from mis_dro.dataset import *
from mis_dro.npl import *
from mis_dro.likelihood import *
from bayesian_dro.Bayesian_DRO_continuous import DGP_STD_TRUNCATED_NORMAL
from mis_dro.gaussian_kernel import *
from mis_dro.bayes_conjugates import *
from scipy.spatial import ConvexHull, convex_hull_plot_2d



def approx_mmd(sample1, sample2):
    m = len(sample1)
    n = len(sample2)
    
    sample1 = sample1.reshape((m,1))
    sample2 = sample2.reshape((n,1))
    
    l = np.sqrt((1/2)*np.median(distance.cdist(sample2, sample2, 'sqeuclidean')))
    
    kyy = k_jax(sample1, sample1, l)
    kxy = k_jax(sample1, sample2, l)
    kxx = k_jax(sample2, sample2, l)

   # first sum
    diag_elements = jnp.diag_indices_from(kyy)
    kyy = kyy.at[diag_elements].set(jnp.repeat(0, m))
    sum1 = jnp.sum(kyy)

    # second sum
    sum2 = jnp.sum(kxy)

    # third sum
    diag_elements = jnp.diag_indices_from(kxx)
    kxx = kxx.at[diag_elements].set(jnp.repeat(0, n))
    sum3 = jnp.sum(kxx)

    return (
        np.sqrt((1 / (m * (m - 1))) * sum1
        - (2 / (n * m)) * sum2
        + (1 / (n * (n - 1))) * sum3)
    )

def approx_kl(mu1, std1, mu2, std2):
    return(np.log(std2/std1) +
           std1**2/(2*std2**2) +
           (mu1 - mu2)**2/(2*std2**2) -
           0.5)

    
if __name__ == "__main__":
    contamination = 0.2
    num_observations = 100
    num_posterior_samples = 100
    num_likelihood_samples = 500
    likelihood = "normal"
    contam_data = sample_dgp("contaminated_normal", num_observations, contamination)
    non_contam_data = sample_dgp("normal", num_observations)

    generator = np.random.default_rng(seed=0)
    theta_sample_mmd_cont = sample_npl(
        contam_data,
        "npl_mmd",
        likelihood,
        num_posterior_samples,
        seed=0,
        lengthscale=-1,
        generator=generator,
    )
    theta_sample_mmd_noncont = sample_npl(
        non_contam_data,
        "npl_mmd",
        likelihood,
        num_posterior_samples,
        seed=0,
        lengthscale=-1,
        generator=generator,
    )

    posterior = "normal_gamma"
    likelihood = "normal"
    dim = 1
    theta_prior = default_prior_params(posterior)
    theta_posterior = get_posterior_params(posterior, non_contam_data, theta_prior)
    theta_sample_kl = sample_posterior(posterior, theta_posterior, num_likelihood_samples, generator=generator)
    theta_pp = posterior_predictive_params(posterior, theta_posterior)
    pp_sample_kl = sample_posterior_predictive(likelihood, posterior, theta_pp, dim, num_likelihood_samples, generator)
    mean_range = np.linspace(0, 50, num=100)
    std_range = np.linspace(0.001, 50, num=100)
        
    cmap = plt.get_cmap('RdYlBu')
    for i, data in enumerate([non_contam_data, contam_data]):
        pairs_inside_RoBAS = []
        pairs_inside_BASPE = []
        pairs_inside_BASPP = []
        if i == 1:
            dataset_name = "contaminated data"
            theta_sample_mmd = theta_sample_mmd_cont
        else:
            dataset_name = "non-contaminated data"
            theta_sample_mmd = theta_sample_mmd_noncont
        
        posterior = "normal_gamma"
        likelihood = "normal"
        dim = 1 
        theta_prior = default_prior_params(posterior)
        theta_posterior = get_posterior_params(posterior, data, theta_prior)
        theta_sample_kl = sample_posterior(posterior, theta_posterior, num_likelihood_samples, generator=generator)
        theta_pp = posterior_predictive_params(posterior, theta_posterior)
        pp_sample_kl = sample_posterior_predictive(likelihood, posterior, theta_pp, dim, num_likelihood_samples, generator)
        mu_pp, scale_pp, dof = theta_pp[0]
        q = sp.stats.t(df=dof, loc=mu_pp, scale=scale_pp)

        for mu, std in tqdm(itertools.product(mean_range, std_range)):
            # sample from P
            samples_from_P = generator.normal(
                loc=mu,
                scale=std,
                size=num_likelihood_samples,
            ) 
            
            # for each theta calculate approximate MMD
            epsilon = 0.35
            mmds = np.zeros(num_posterior_samples)
            for j in range(num_posterior_samples):
                samples_from_Ptheta = generator.normal(
                    loc=theta_sample_mmd[j,0],
                    scale=theta_sample_mmd[j,1],
                    size=num_likelihood_samples,
                )
                
                mmds[j] = approx_mmd(samples_from_P, samples_from_Ptheta)
            exp_mmd = mmds.mean()
            if exp_mmd < epsilon:
                pairs_inside_RoBAS.append((mu,std))
                
            # for each theta calculate approximate KL
            epsilon = 0.35
            kls = np.zeros(num_posterior_samples)
            for j in range(num_posterior_samples):
                kls[j] = approx_kl(mu, std, theta_sample_kl[j,0], theta_sample_kl[j,1])
                
            exp_kl = kls.mean()
            if exp_kl < epsilon:
                pairs_inside_BASPE.append((mu,std))

            # PP
            p = sp.stats.norm(loc=mu, scale=std)
            kl = np.mean(np.log(p.pdf(samples_from_P) / q.pdf(samples_from_P)))
            if kl < epsilon:
                pairs_inside_BASPP.append((mu,std))

        if i == 0:
            non_contam_pairs_BASPE = np.array(pairs_inside_BASPE)
            non_contam_pairs_BASPP = np.array(pairs_inside_BASPP)
            non_contam_pairs_RoBAS = np.array(pairs_inside_RoBAS)

        else:
            contam_pairs_BASPE = np.array(pairs_inside_BASPE)
            contam_pairs_BASPP = np.array(pairs_inside_BASPP)
            contam_pairs_RoBAS = np.array(pairs_inside_RoBAS)
        # print(pairs_RoBAS)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharex=True, sharey=True)

    ax = axes[0]
    pairs_BASPE = non_contam_pairs_BASPE
    pairs_BASPP = non_contam_pairs_BASPP
    pairs_RoBAS = non_contam_pairs_RoBAS

    hull = sp.spatial.ConvexHull(pairs_BASPE)
    xy = np.array([pairs_BASPE[hull.vertices, 0], pairs_BASPE[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#4daf4a")
    patch = ax.add_patch(polygon)
    ax.scatter(pairs_BASPE[:,0], pairs_BASPE[:, 1], alpha=0.3, marker="x", color="#4daf4a", s=6)
    ax.scatter(25, 10, color="black", alpha=1, marker="o")
    # ax.set_title(f'BAS with {dataset_name} dataset')
    ax.set_ylabel('standard deviation')

    hull = sp.spatial.ConvexHull(pairs_BASPP)
    xy = np.array([pairs_BASPP[hull.vertices, 0], pairs_BASPP[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#984ea3")
    pp_patch = ax.add_patch(polygon)
    ax.scatter(pairs_BASPP[:,0], pairs_BASPP[:, 1], alpha=0.3, marker="x", color="#984ea3", s=6)
    ax.scatter(25, 10, color="black", alpha=1, marker="o")
    # ax.set_title(f'BAS with {dataset_name} dataset')
    # ax.set_ylabel('standard deviation')

    hull = sp.spatial.ConvexHull(pairs_RoBAS)
    xy = np.array([pairs_RoBAS[hull.vertices, 0], pairs_RoBAS[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#377eb8")
    ro_patch = ax.add_patch(polygon)
    ax.scatter(pairs_RoBAS[:,0], pairs_RoBAS[:, 1], alpha=0.3, marker="s", s=6)
    dgp = ax.scatter(25, 10, color="black", alpha=1, marker="o")
    ax.set_title('Non-contaminated dataset', size=14)
    ax.set_xlabel('mean', size=14)
    ax.set_ylabel('standard deviation', size=14)
    ax.legend([patch, pp_patch, ro_patch, dgp], ["BAS$_{PE}$", "BAS$_{PP}$", "RoBAS", "DGP"], loc='upper right', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    ax = axes[1]
    pairs_BASPE = contam_pairs_BASPE
    pairs_BASPP = contam_pairs_BASPP
    pairs_RoBAS = contam_pairs_RoBAS

    hull = sp.spatial.ConvexHull(pairs_BASPE)
    xy = np.array([pairs_BASPE[hull.vertices, 0], pairs_BASPE[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#4daf4a")
    patch = ax.add_patch(polygon)
    ax.scatter(pairs_BASPE[:,0], pairs_BASPE[:, 1], alpha=0.3, marker="x", color="#4daf4a", s=6)
    ax.scatter(25, 10, color="black", alpha=1, marker="o")
    # ax.set_title(f'BAS with {dataset_name} dataset')
    # ax.set_ylabel('standard deviation', size=12)

    hull = sp.spatial.ConvexHull(pairs_BASPP)
    xy = np.array([pairs_BASPP[hull.vertices, 0], pairs_BASPP[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#984ea3")
    patch = ax.add_patch(polygon)
    ax.scatter(pairs_BASPP[:,0], pairs_BASPP[:, 1], alpha=0.3, marker="x", color="#984ea3", s=6)
    ax.scatter(25, 10, color="black", alpha=1, marker="o")

    hull = sp.spatial.ConvexHull(pairs_RoBAS)
    xy = np.array([pairs_RoBAS[hull.vertices, 0], pairs_RoBAS[hull.vertices, 1]]).T
    polygon = mpl.patches.Polygon(xy, alpha=0.5, color="#377eb8")
    ro_patch = ax.add_patch(polygon)
    ax.scatter(pairs_RoBAS[:,0], pairs_RoBAS[:, 1], alpha=0.3, marker="s", s=6)
    dgp = ax.scatter(25, 10, color="black", alpha=1, marker="o")
    ax.set_title('Contaminated dataset', size=14)
    ax.set_xlabel('mean', size=14)
    ax.tick_params(axis='both', labelsize=14)
    # ax.set_ylabel('standard deviation')
    # ax.legend([patch, ro_patch, dgp], ["BAS", "RoBAS", "DGP"], loc='upper left')

    # plt.tight_layout()
    fig.savefig("./misdro/intro_fig_100.pdf", bbox_inches="tight")