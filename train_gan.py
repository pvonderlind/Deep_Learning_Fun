import torch.cuda

from GAN import GAN, img_to_vector, vec_to_img, noise
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd.variable import Variable
from logger import Logger

from datetime import datetime
import os


def init_mnist():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    outdir = './dataset'
    return datasets.MNIST(root=outdir, train=True, transform=compose, download=True)


def init_celeba():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    outdir = './dataset'
    return datasets.CelebA(root=outdir, split="train", transform=compose, download=True)


class GanRunner:

    def __init__(self, model_type, dataset_name, latent_size, batch_size, n_episodes):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.n_episodes = n_episodes

        self.img_size = 64
        self.image_channels = 1
        self.dataset = self.load_dataset()

        self.gan = self.init_gan_models()

        self.logger = Logger(self.model_type, self.dataset_name)

    def load_dataset(self):
        return {
            'mnist': init_mnist(),
            'celeba': init_celeba()
        }.get(self.dataset_name, 'mnist')

    def init_mnist(self):
        self.image_channels = 1
        self.img_size = 64
        compose = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        outdir = './dataset'
        return datasets.MNIST(root=outdir, train=True, transform=compose, download=True)

    def init_celeba(self):
        self.image_channels = 3
        self.img_size = 64
        compose = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        outdir = './dataset'
        return datasets.CelebA(root=outdir, split="train", transform=compose, download=True)

    def init_gan_models(self):
        return {
            'DCGAN': GAN(use_deep=True),
            'GAN': GAN()
        }.get(self.model_type, "DCGAN")

    def train_gan(self):
        self.print_run_info()

        try:
            for episode in range(n_episodes):
                for n_batch, (real_batch, _) in enumerate(self.dataset):
                    randnoise = self.generate_noise()

                    if not deep:
                        real_batch = Variable(img_to_vector(real_batch))

                    if torch.cuda.is_available():
                        real_batch = real_batch.cuda()
                        randnoise = randnoise.cuda()

                    # Generate fake data from generator
                    # Detach since we don't want to gradient descent when generating
                    # our fake data for the discriminator.
                    fake_batch = self.gan.generator(randnoise).detach()

                    disc_error, gen_error, disc_pred_real, disc_pred_fake = self.perform_model_update(fake_batch,
                                                                                                      real_batch)

                    self.logger.log(disc_error, gen_error, episode, n_batch, num_batches)

                    if n_batch % 100 == 0:
                        if not deep:
                            test_imgs = vec_to_img(self.gan.generator(randnoise)).data.cpu()
                        else:
                            test_imgs = self.gan.generator(randnoise).data.cpu()

                        self.logger.log_images(test_imgs, num_test_samples, episode, n_batch, num_batches)

                        self.logger.display_status(episode, n_episodes, n_batch, num_batches, disc_error,
                                                   gen_error, disc_pred_real, disc_pred_fake)
        except KeyboardInterrupt:
            print(f"Closing early after keyboard interrupt.")

        print(f"Saving model before closing ...")
        fname = datetime.now().strftime(f'%d-%m-%y-%Hh_%Mm_{model_name}_weights.pth')
        path = f'./models/{model_name}/{dataset_name}/{fname}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.gan.generator.state_dict(), path)

    def generate_noise(self):
        return {
            "DCGAN": torch.randn((self.batch_size, self.latent_size, 1, 1)),
            "GAN": Variable(torch.randn(self.batch_size, self.latent_size))
        }.get(self.model_type, "DCGAN")

    def perform_model_update(self, fake_batch, real_batch):
        disc_error, disc_pred_real, disc_pred_fake = self.gan.train_discriminator(real_batch, fake_batch)
        # Train generator --> Gradient desc this time when generating fake data!
        fake_batch = self.gan.generator(randnoise)
        gen_error = self.gan.train_generator(fake_batch)
        return disc_error, gen_error, disc_pred_real, disc_pred_fake

    def print_run_info(self):
        print(
            f"Running {self.model_type} using parameters batchsize:{self.batch_size}, latentsize:{self.latent_size} on {self.dataset_name}")
        print(f"Training for {self.n_episodes} episodes with batchsize {self.batch_size}")
        print(f"Running on CUDA GPU: {torch.cuda.is_available()}")


# TODO: Implement Argparser to control hyperparameters, dataset etc.
if __name__ == "__main__":
    deep = True
    batch_size = 128
    latent_size = 100
    gan = GAN(use_deep=deep, image_size=64, image_channels=1)
    n_episodes = 200
    data = init_mnist()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    model_name = "DCGAN"
    dataset_name = "mnist_64"
    logger = Logger(model_name, dataset_name)
    num_batches = len(dataloader)

    num_test_samples = 16
    randnoise = noise(num_test_samples)

    if deep:
        print(f"Running DCGAN using parameters batchsize:{batch_size}, latentsize:{latent_size} on MNIST")
    else:
        print(f"Running GAN using parameters batchsize:{batch_size}, latentsize:{latent_size} on MNIST")
    print(f"Trainign for {n_episodes} episodes with batchsize {128}")
    print(f"Running on CUDA GPU: {torch.cuda.is_available()}")

    try:
        for episode in range(n_episodes):
            for n_batch, (real_batch, _) in enumerate(dataloader):
                if deep:
                    randnoise = torch.randn((batch_size, latent_size, 1, 1))
                else:
                    N = real_batch.size(0)
                    real_batch = Variable(img_to_vector(real_batch))
                    randnoise = noise(N)

                if torch.cuda.is_available():
                    real_batch = real_batch.cuda()
                    randnoise = randnoise.cuda()

                # Generate fake data from generator
                # Detach since we don't want to gradient descent when generating
                # our fake data for the discriminator.
                fake_data = gan.generator(randnoise).detach()

                # Train discriminator
                disc_error, disc_pred_real, disc_pred_fake = gan.train_discriminator(real_batch, fake_data)

                # Train generator --> Gradient desc this time when generating fake data!
                fake_data = gan.generator(randnoise)

                gen_error = gan.train_generator(fake_data)

                logger.log(disc_error, gen_error, episode, n_batch, num_batches)

                if n_batch % 100 == 0:
                    if not deep:
                        test_imgs = vec_to_img(gan.generator(randnoise)).data.cpu()
                    else:
                        test_imgs = gan.generator(randnoise).data.cpu()

                    logger.log_images(test_imgs, num_test_samples, episode, n_batch, num_batches)

                    logger.display_status(episode, n_episodes, n_batch, num_batches, disc_error,
                                          gen_error, disc_pred_real, disc_pred_fake)
    except KeyboardInterrupt:
        print(f"Closing early after keyboard interrupt.")

    print(f"Saving model before closing ...")
    fname = datetime.now().strftime(f'%d-%m-%y-%Hh_%Mm_{model_name}_weights.pth')
    path = f'./models/{model_name}/{dataset_name}/{fname}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(gan.generator.state_dict(), path)
