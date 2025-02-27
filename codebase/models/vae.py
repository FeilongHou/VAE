import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        qm, qv = self.enc.encode(x)
        z = ut.sample_gaussian(qm, qv)
        x_reconstruct = self.sample_x_given(z)
        #x_reconstruct = self.dec.decode(z)
        
        pv, pm = self.z_prior_v, self.z_prior_m
        #rec = F.binary_cross_entropy(x_reconstruct, x, reduction='sum')
        #kl = torch.mean(-0.5 * torch.sum(1 + qv - qm ** 2 - qv.exp(), dim = 1), dim = 0)

        kl = ut.kl_normal(qm, qv, pm, pv)
        kl = kl.mean(dim=0)
        rec = ut.log_bernoulli_with_logits(x, x_reconstruct)
        #rec = -torch.log(x_reconstruct)
        rec = rec.mean(dim=0)
        
        nelbo = kl + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def generate(self, batch=200):
        """
        Generate MINIST images default 200

        Args:
            batch: int: num of generated images
        """
        x = self.sample_x(batch)
        x = x.cpu().detach().numpy()
        fig, axes = plt.subplots(10, 20, figsize=(15,15))
        axes = axes.flatten()
        for i, tensor in enumerate(x):
            tensor = tensor.reshape(28,28)
            axes[i].imshow(tensor, cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        fig.suptitle(f"{batch} generated images")
        plt.show()


    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
