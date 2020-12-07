import numpy as np
import torch 
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os
import sys
import json
import time

from scaler import *
from opts import parse_opts
from loss_funcs import *

# lightweight GAN model
from lwgan.lightweight_gan import LightweightGAN

device = torch.device("cuda")
    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(opt)
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')

    # prepare scaler for data
    if opt.dataset == 'radarJMA':
        if opt.data_scaling == 'linear':
            scl = LinearScaler()

    # define model

    # Data Parallel Multi-GPU Run
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model) # make parallel
    model.to(device)
    
    if not opt.no_train:
        # prepare transform
        if opt.aug_rotate > 0.0:
            Rot = RandomRotateVideo(degrees=opt.aug_rotate)
            Resize = RandomResizeVideo(factor=opt.aug_resize)
            composed = transforms.Compose([Rot,Resize])
        else:
            composed = None
        # loading datasets
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
            valid_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=False)
        
        #dd = next(iter(train_dataset))
    
        modelinfo.write('Model Structure \n')
        modelinfo.write(str(model))
        count_parameters(model,modelinfo)
        modelinfo.close()
        

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)


class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        optimizer="adam",
        latent_dim = 256,
        image_size = 128,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        gp_weight = 10,
        gradient_accumulate_every = 1,
        attn_res_layers = [],
        sle_spatial = False,
        disc_output_size = 5,
        antialias = False,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 1.,
        save_every = 1000,
        evaluate_every = 1000,
        trunc_psi = 0.6,
        aug_prob = None,
        aug_types = ['translation', 'cutout'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        amp = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.config_path = self.models_dir / name / '.config.json'

        assert is_power_of_two(image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        self.optimizer = optimizer
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.generator_top_k_gamma = 0.99
        self.generator_top_k_frac = 0.5

        self.attn_res_layers = attn_res_layers
        self.sle_spatial = sle_spatial
        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.syncbatchnorm = is_ddp

        self.amp = amp
        self.G_scaler = None
        self.D_scaler = None
        if self.amp:
            self.G_scaler = GradScaler()
            self.D_scaler = GradScaler()

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when
        # switching from multi-gpu back to single gpu

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr = self.lr,
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            sle_spatial = self.sle_spatial,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            rank = self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.sle_spatial = config.pop('sle_spatial', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'sle_spatial': self.sle_spatial
        }

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob   = default(self.aug_prob, 0)
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        amp_context = autocast if self.amp else null_context

        def backward(amp, loss, scaler):
            if amp:
                return scaler.scale(loss).backward()
            loss.backward()

        def optimizer_step(amp, optimizer, scaler):
            if amp:
                scaler.step(optimizer)
                scaler.update()
                return
            optimizer.step()

        backward = partial(backward, self.amp)
        optimizer_step = partial(optimizer_step, self.amp)

        # train discriminator
        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()

            with amp_context():
                generated_images = G(latents)
                fake_output, fake_output_32x32, _ = D_aug(generated_images.detach(), detach = True, **aug_kwargs)

                real_output, real_output_32x32, real_aux_loss = D_aug(image_batch,  calc_aux_loss = True, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                divergence = hinge_loss(real_output_loss, fake_output_loss)
                divergence_32x32 = hinge_loss(real_output_32x32, fake_output_32x32)
                disc_loss = divergence + divergence_32x32

                aux_loss = real_aux_loss
                disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]
                outputs = list(map(self.D_scaler.scale, outputs)) if self.amp else outputs

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                       grad_outputs=list(map(lambda t: torch.ones(t.size(), device = image_batch.device), outputs)),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = (1. / self.D_scaler.get_scale()) if self.amp else 1.
                gradients = scaled_gradients * inv_scale

                with amp_context():
                    gradients = gradients.reshape(batch_size, -1)
                    gp =  self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    if not torch.isnan(gp):
                        disc_loss = disc_loss + gp
                        self.last_gp_loss = gp.clone().detach().item()

            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            backward(disc_loss, self.D_scaler)
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        optimizer_step(self.GAN.D_opt, self.D_scaler)

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            with amp_context():
                generated_images = G(latents)
                fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
                fake_output_loss = fake_output.mean(dim = 1) + fake_output_32x32.mean(dim = 1)

                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

                loss = fake_output_loss.mean()
                gen_loss = loss

                gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backward(gen_loss, self.G_scaler)
            total_gen_loss += loss 

        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        optimizer_step(self.GAN.G_opt, self.G_scaler)

        # calculate moving averages

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch = next(self.loader)
            for k in range(real_batch.size(0)):
                torchvision.utils.save_image(real_batch[k, :, :, :], real_path + '{}.png'.format(k + batch_num * self.batch_size))

        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.GE, latents)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :], str(Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)

    @torch.no_grad()
    def generate_truncated(self, G, style, trunc_psi = 0.75, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_truncated(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.amp:
            save_data = {
                **save_data,
                'G_scaler': self.G_scaler.state_dict(),
                'D_scaler': self.D_scaler.state_dict()
            }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

        if self.amp:
            if 'G_scaler' in load_data:
                self.G_scaler.load_state_dict(load_data['G_scaler'])
            if 'D_scaler' in load_data:
                self.D_scaler.load_state_dict(load_data['D_scaler'])
