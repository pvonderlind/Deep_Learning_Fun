import torch
from torch import nn, optim
from torch.autograd.variable import Variable


def img_to_vector(image):
    return image.view(image.size(0), 784)


def vec_to_img(vector):
    return vector.view(vector.size(0), 1, 28, 28)


def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


def ones_target(size):
    target = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return target.cuda()
    return target


def zeros_target(size):
    target = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return target.cuda()
    return target


class Discriminator(torch.nn.Module):

    def __init__(self, dim_input=784, dim_out=1):
        super(Discriminator, self).__init__()
        n_features = dim_input  # MNIST dimensions
        n_out = dim_out

        self.input = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # Latent representations of the input image.
        n_features = 100
        n_out = 784

        self.input = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


class DCGenerator(torch.nn.Module):

    def __init__(self, latent_size=100, img_channels=3, feature_map_size=64):
        super(DCGenerator, self).__init__()

        self.main = nn.Sequential(
            # First layer, encoded latent space Z going directly into first convolutional layer and batch-normalization.
            self.conv_block(latent_size, feature_map_size * 16, 4, 1, 0),  # (64 * 16) x 4 x 4 = 1024 x 4 x 4
            self.conv_block(feature_map_size * 16, feature_map_size * 8, 4, 2, 1),  # (64 * 8) x 8 x 8 = 512 x 8 x 8
            self.conv_block(feature_map_size * 8, feature_map_size * 4, 4, 2, 1),  # (64 * 4) x 16 x 16 = 256 x 16 x 16
            self.conv_block(feature_map_size * 4, feature_map_size * 2, 4, 2, 1),  # (64 * 2) x 32 x 32 = 128 x 32 x 32
            nn.ConvTranspose2d(feature_map_size * 2, img_channels, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.Tanh()
            # Output size final (img_channels) x 64 x 64
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)


class DCDiscriminator(torch.nn.Module):

    def __init__(self, img_channels=3, feature_map_size=64):
        super(DCDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_size, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2),
            self.conv_block(feature_map_size, feature_map_size * 2, 4, 2, 1),
            self.conv_block(feature_map_size * 2, feature_map_size * 4, 4, 2, 1),
            self.conv_block(feature_map_size * 4, feature_map_size * 8, 4, 2, 1),
            nn.Conv2d(feature_map_size * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.main(x)


class GAN:

    def __init__(self, learning_rate=0.0002, beta1=0.5, use_deep=True, image_channels=3, image_size=64):
        if use_deep:
            self.generator = DCGenerator(feature_map_size=image_size, img_channels=image_channels)
            self.generator.apply(self.dc_weight_init)
            self.discriminator = DCDiscriminator(feature_map_size=image_size, img_channels=image_channels)
            self.discriminator.apply(self.dc_weight_init)
        else:
            self.generator = Generator()
            self.discriminator = Discriminator()

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        self.lr = learning_rate

        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.generator_opt = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(beta1, 0.999))

        self.loss = nn.BCELoss()

    @staticmethod
    def dc_weight_init(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def train_discriminator(self, real_data, fake_data):
        N = real_data.size(0)

        # Reset gradient to zero
        self.discriminator_opt.zero_grad()

        real_prediction = self.discriminator(real_data)
        real_error = self.loss(real_prediction, torch.ones_like(real_prediction))
        real_error.backward()

        fake_prediction = self.discriminator(fake_data)
        fake_error = self.loss(fake_prediction, torch.zeros_like(fake_prediction))
        fake_error.backward()

        self.discriminator_opt.step()

        return real_error + fake_error, real_prediction, fake_prediction

    def train_generator(self, fake_data):
        N = fake_data.size(0)

        self.generator_opt.zero_grad()

        prediction = self.discriminator(fake_data)

        # Error and back propagation
        error = self.loss(prediction, torch.ones_like(prediction))
        error.backward()

        # Update weights
        self.generator_opt.step()

        return error
