import argparse
import torch
from torch import optim, nn
import numpy as np
from Dataset import Dataset
import models

def add_noise(std, batch,device='cpu'):
    x = batch
    noise = np.random.normal(0, std, size=x.shape)
    noise = torch.from_numpy(noise).float()
    x_noisy = (x + noise).clamp(0.0, 1.0)

    return x.to(device), x_noisy.to(device)


def train_GAN(generator, discriminator,dataloader,device='cpu',lambda_ = 0.4, discriminator_iter= 1,std=0.155**2):
    generator.train()
    discriminator.train()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    criterion_gen = nn.MSELoss()
    discriminator_iterations = discriminator_iter
    for epoch in range(2):
        for iteration, (inputs, labels) in enumerate(dataloader):
            #Fix Generator and train discriminator
            for iteration2, (inputs2, labels2) in enumerate(dataloader):
                if iteration2 >= discriminator_iterations:
                    break
                # update discriminator
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                #Add noise for robustness
                x, x_noisy = add_noise(std,inputs2,device)
                out_generator = generator(x_noisy)
                #Calculate loss of discriminator on generated input
                discr_generator = discriminator(out_generator)
                zeros = torch.zeros(len(discr_generator), dtype=torch.long, device=device)
                loss_discr_generated = criterion(discr_generator,zeros)
                #Calculate loss of discriminator on real input
                discr_original = discriminator(x)
                ones = torch.ones(len(discr_original), dtype=torch.long, device=device)
                loss_discr_original = criterion(discr_original, ones)

                #Sum discriminator loss
                loss_discr = loss_discr_generated + loss_discr_original

                #backpropagation
                loss_discr.backward()
                optimizerD.step()
                if iteration%10==0:
                    print('Total Discriminator Loss %f Iteration %d '%(loss_discr,iteration))


            #Fix Discriminator and Train Generator
            # Add noise for robustness
            x, x_noisy = add_noise(std,inputs, device)
            out_generator = generator(x_noisy)
            #Get discriminator's decision
            discrimator_decision_generated = discriminator(out_generator)
            ones = torch.ones(len(discrimator_decision_generated), dtype=torch.long, device=device)
            #Loss from min-max game
            loss_generator = criterion(discrimator_decision_generated, ones)
            #Loss from Encoder-Decoder module
            reconstruction_loss = criterion_gen(out_generator,x)
            #Loss Combination
            generator_loss = loss_generator + lambda_ *reconstruction_loss
            #Backpropagtion
            generator_loss.backward()
            optimizerG.step()
            if iteration%10 == 0:
                print('Total Generator Loss %f Iteration %d ' % (generator_loss, iteration))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, default = '../../Data/test')
    parser.add_argument("--test_dir", required=False, default = '../../Data/testset')
    parser.add_argument("--checkpoint", required=False,default = 'model.pt')

    args = parser.parse_args()

    training_dir = args.data_dir
    test_dir = args.test_dir

    image_datasets = {'train': Dataset(training_dir,mode='positive')}

    target_dataloader = { x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=1) for x in ['train']}
    sigma = 0.155
    model = models.NoveltyDetector(noise_std=sigma**2)

    train_GAN(model.Generator, model.Discriminator,target_dataloader['train'],device='cpu',lambda_ = 0.4, discriminator_iter= 1)