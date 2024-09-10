import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from models.env import AttrDict, build_env
from audiotools import AudioSignal

from models.model.discriminator import Discriminator

from models.model.dac_vae import DACVAE
import modules.loss as losses 

from utils import plot_spectrogram
from utils import scan_checkpoint
from utils import load_checkpoint
from utils import save_checkpoint

# from data.dataset_libritts import LIBRITTS_Duration
from data.dataset import dataset_slakh2100

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))


    # generator = DACVAE(
    #     encoder_dim = h.encoder_dim,
    #     encoder_rates = h.encoder_rates,
    #     latent_dim = h.latent_dim,
    #     decoder_dim = h.decoder_dim,
    #     decoder_rates = h.decoder_rates,
    #     sample_rate = h.sample_rate).to('cpu')

    generator = DACVAE(
        encoder_dim = 64,
        encoder_rates = [2, 4, 5, 8],
        latent_dim = 64,
        decoder_dim = 1536,
        decoder_rates = [8, 5, 4, 2],
        sample_rate = 22050).to(device)

    
    discriminator = Discriminator(
        sample_rate=h.D_sample_rate,
        rates=h.D_rates,
        periods=h.D_periods,
        fft_sizes=h.D_fft_sizes,
        bands=h.D_bands
    ).to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])

        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        # steps = 0
        # last_epoch = -1

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.lr,
        betas=h.betas)
    optim_d = torch.optim.Adam(
        discriminator.parameters(),
        h.lr,
        betas=h.betas)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.ExponentialLR_gamma)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.ExponentialLR_gamma)
    

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss(
        window_lengths=h.msSTFTLoss_window_lengths
    )
    mel_loss = losses.MelSpectrogramLoss(
        n_mels=h.n_mels,
        window_lengths=h.window_lengths,
        mel_fmin=h.mel_fmin,
        mel_fmax=h.mel_fmax,
        pow=h.pow,
        clamp_eps=h.clamp_eps,
        mag_weight=h.mag_weight

    )
    gan_loss = losses.GANLoss(discriminator)

    trainset = dataset_slakh2100(
        meta_data_path=a.slakh_train_metadata_path, 
        sample_rate=h.sampling_rate,
        segment_length=h.segment_length,
        shuffle=True
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = dataset_slakh2100(
            meta_data_path=a.slakh_validation_metadata_path, 
            sample_rate=h.sampling_rate,
            segment_length=22050*5,
            shuffle=True
        )
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    lambdas = h.lambdas
    # scaler = torch.cuda.amp.GradScaler()


    for epoch in range(max(0, last_epoch), a.training_epochs):
        generator.train()
        discriminator.train()
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            # if i > 10:
            #     break
            if rank == 0:
                start_b = time.time()

            output = {}

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            input = batch.unsqueeze(1)
            input = torch.autograd.Variable(input.to(device, non_blocking=True))

            signal = AudioSignal(input, h.sampling_rate)
            out = generator(signal.audio_data, signal.sample_rate)

            recons = AudioSignal(out["audio"], signal.sample_rate)
            kl_loss = out['loss_KLD'].mean()
            # commitment_loss = out["vq/commitment_loss"]
            # codebook_loss = out["vq/codebook_loss"]

            output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)

            optim_d.zero_grad()
            output["adv/disc_loss"].backward()
            output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), 10.0
            )
            optim_d.step()
            scheduler_d.step()

            output["stft/loss"] = stft_loss(recons, signal)
            output["mel/loss"] = mel_loss(recons, signal)
            output["waveform/loss"] = waveform_loss(recons, signal)
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = gan_loss.generator_loss(recons, signal)
            # output["vq/commitment_loss"] = commitment_loss
            # output["vq/codebook_loss"] = codebook_loss
            output['loss_KLD'] = kl_loss
            output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

            optim_g.zero_grad()
            output["loss"].backward()
            # scaler.scale(output["loss"]).backward()
            output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), 1e3
            )

            optim_g.step()
            # scaler.step(optim_g)
            # scaler.update()
            scheduler_g.step()

             
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    result_dict = {}
                    for key in output.keys():
                        result_dict[key] = output[key].item()
                    print('steps: ', steps, result_dict, time.time()-start_b)
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': (generator.module if h.num_gpus > 1
                                          else generator).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'discriminator': (discriminator.module
                                    if h.num_gpus > 1 else discriminator).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'scheduler_g':
                            scheduler_g.state_dict(),
                            'scheduler_d':
                            scheduler_d.state_dict(),      
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    for key in output.keys():
                        sw.add_scalar(key, output[key].item(), steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_dict = {
                        "mel/loss": 0,
                        "stft/loss": 0,
                        "waveform/loss": 0,
                        'loss_KLD': 0
                    }
                    with torch.no_grad():
                        print('validation---------------------------------------')
                        for j, batch in enumerate(validation_loader):
                            if j > 200:
                                break

                            input = batch.unsqueeze(1)
                            input = input[..., :22050*5]
                            input = torch.autograd.Variable(input.to(device, non_blocking=True))

                            signal = AudioSignal(input, h.sampling_rate)


                            out = generator(signal.audio_data, signal.sample_rate)
                            recons = AudioSignal(out["audio"], signal.sample_rate)


                            val_err_dict["mel/loss"] += mel_loss(recons, signal).item()
                            val_err_dict["stft/loss"] += stft_loss(recons, signal).item()
                            val_err_dict["waveform/loss"] += waveform_loss(recons, signal).item()
                            val_err_dict["loss_KLD"] += out['loss_KLD'].mean().clone().item()

                            if j <= 20:
                                sw.add_audio('gt/y_{}'.format(j), input[0],
                                                steps, h.sampling_rate)
                                sw.add_audio('generated/y_hat_{}'.format(j),
                                             out["audio"][0], steps, h.sampling_rate)

                        for key in val_err_dict:
                            val_err_dict[key] = val_err_dict[key] / (j + 1)
                            sw.add_scalar("validation/mel_spec_error", val_err_dict[key], steps)

            steps += 1


        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    # parser.add_argument('--root_libritts', required=True)
    parser.add_argument('--slakh_train_metadata_path', required=True)
    parser.add_argument('--slakh_validation_metadata_path', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=20, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, ))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()