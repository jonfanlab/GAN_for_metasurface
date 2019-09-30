import os
import logging
from tqdm import tqdm

from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F 
import torch
import utils
import scipy.io as io


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def visualize_training_generator(generator, fig_path, cuda=False, n_row = 4, n_col = 4):
    generator.eval()
    wavelengths = torch.linspace(-1, 1, n_col).view(1, n_col).repeat(n_row, 1).view(-1, 1)
    angles = torch.linspace(-1, 1, n_row).view(n_row, 1).repeat(1, n_col).view(-1, 1)
    labels = torch.cat([wavelengths, angles], -1).type(Tensor)
    imgs, _ = sample_images(generator, labels, cuda)

    paddings = (0, 0, 0, imgs.size(2)-1)
    imgs = F.pad(imgs, paddings, mode='reflect')
    save_image(imgs, fig_path, n_row)
    generator.train()


def sample_images(generator, labels, cuda=False):   
    if cuda:
        z = Variable(torch.cuda.FloatTensor(labels.size(0), generator.noise_dim).normal_())
        z.cuda()
    else:
        z = Variable(torch.randn(labels.size(0), generator.noise_dim))        
    return generator(labels, z), z


def evaluate(generator, wavelengths, angles, num_imgs, params):
    generator.eval()
    for wavelength in wavelengths:
        for angle in angles:
            filename = 'ccGAN_imgs_Si_w' + str(wavelength) +'_' + str(angle) +'deg.mat'
            mdict = {'wavelength': wavelength, 'angle': angle}

            w = (wavelength - params.wc)/params.wspan
            theta = (angle - params.ac)/params.aspan

            labels = Tensor([w, theta]).repeat(num_imgs, 1)
            images, noise = sample_images(generator, labels, params.cuda)

            mdict['imgs'] = torch.squeeze(images).cpu().detach().numpy()
            mdict['noise'] = noise.data.cpu().numpy()

            file_path = os.path.join(params.output_dir,'outputs',filename)
            io.savemat(file_path, mdict=mdict)

        logging.info('wavelength = '+str(wavelength)+ ' is done. \n')


def compute_gradient_penalty(D, real_samples, fake_samples, labels, cuda=False):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).type(Tensor)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).type(Tensor).requires_grad_(True)
    d_interpolates = D(interpolates, labels)

    fake = Variable(Tensor(real_samples.size(0), 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(models, optimizers, dataloader, params):

    generator, discriminator = models
    optimizer_G, optimizer_D = optimizers

    generator.train()
    discriminator.train()

    gen_loss_history = []
    dis_loss_history = []

    with tqdm(total=params.numIter) as t:
        it = 0
        while True:
            for i, (real_imgs, labels) in enumerate(dataloader):

                it +=1 
                if it > params.numIter:
                    model_dir = os.path.join(params.output_dir, 'model')
                    utils.save_checkpoint({'iter': it,
                                           'gen_state_dict': generator.state_dict(),
                                           'dis_state_dict': discriminator.state_dict(),
                                           'optim_G': optimizer_G.state_dict(),
                                           'optim_D': optimizer_D.state_dict()},
                                           checkpoint=model_dir)
                    return (gen_loss_history, dis_loss_history)

                # move to GPU if available
                if params.cuda:
                    real_imgs, labels = real_imgs.cuda(), labels.cuda()
          
                # convert to torch Variables
                Tensor = torch.cuda.FloatTensor if params.cuda else torch.FloatTensor
                real_imgs, labels = Variable(real_imgs.type(Tensor)), Variable(labels.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------
                

                optimizer_D.zero_grad()
                # Sample noise as generator input
                z = Variable(torch.randn(labels.size(0), params.noise_dims).type(Tensor))

                #if params.cuda:
                 #   z.cuda()
                # Generate a batch of images

                fake_imgs = generator(labels ,z)

             
                # Real images
                real_validity = discriminator(real_imgs, labels)

             
                # Fake images
                
                fake_validity = discriminator(fake_imgs, labels)


                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data, params.cuda)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + params.lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                dis_loss_history.append(d_loss.data)
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                 # Train the generator every n_critic steps
                if it % params.n_critic == 0:
                    # Generate a batch of images
                    fake_imgs = generator(labels, z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs, labels)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    gen_loss_history += [g_loss.data] * params.n_critic
                #t.set_postfix(loss='{:05.3f}'.format(g_loss.data))
                #t.update()

                if it % 250 == 0:
                    logging.info('Generator loss: %f' % g_loss.data)
                    logging.info('Discriminator loss: %f' % d_loss.data)
                    fig_path = os.path.join(params.output_dir, 'figures', 'iter{}.png'.format(it))
                    visualize_training_generator(generator, fig_path, params.cuda)
                
                t.set_postfix(loss='{:05.3f}'.format(g_loss.data))
                t.update()



