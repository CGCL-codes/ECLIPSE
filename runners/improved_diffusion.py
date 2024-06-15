import os
import random
import torch
import torchvision.utils as tvu
from improved_diff_utils.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class Diffusion_ddim(torch.nn.Module): 
    def __init__(self, args, config, device=None, model_dir=""):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        path = os.path.join("./diff_ckpt/cifar10", args.sparse_set, args.sparse_diff_model)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)
        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)
        print(len(self.betas))

    def diffusion_purification(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]
            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img
            xs = []

            """adding Gaussian to absorb poison"""
            e = torch.randn_like(x0)
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()


            """denoising process"""
            for i in reversed(range(total_noise_levels)):
                t = torch.tensor([i] * batch_size, device=self.device)
                x = self.diffusion.p_sample(self.model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None)["sample"]
            x0 = x
            xs.append(x0)
            return torch.cat(xs, dim=0)
