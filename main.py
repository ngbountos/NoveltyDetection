import argparse
import torch
from torch import optim, nn
import numpy as np
from Dataset import Dataset
import models
import cv2 as cv

def add_noise(std, batch,device='cpu'):
    x = batch
    noise = np.random.normal(0, std, size=x.shape)
    noise = torch.from_numpy(noise).float()
    x_noisy = (x + noise).clamp(0.0, 1.0)

    return x.to(device), x_noisy.to(device)


def train_GAN(generator, discriminator,dataloader,device='cpu',lambda_ = 0.4, discriminator_iter= 1,std=0.155**2, verbose = False):
    generator.train()
    discriminator.train()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    criterionBCE = nn.BCELoss()
    criterion_gen = nn.MSELoss()
    discriminator_iterations = discriminator_iter
    for epoch in range(30):
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
                discr_generator = discr_generator.view((discr_generator.shape[0]))
                zeros = torch.zeros(len(discr_generator), dtype=torch.float, device=device)

                loss_discr_generated = criterion(discr_generator,zeros)
                #Calculate loss of discriminator on real input
                discr_original = discriminator(x)
                discr_original = discr_original.view((discr_original.shape[0]))

                ones = torch.ones(len(discr_original), dtype=torch.float, device=device)

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
            discrimator_decision_generated = discrimator_decision_generated.view((discrimator_decision_generated.shape[0]))

            ones = torch.ones(len(discrimator_decision_generated), dtype=torch.float, device=device)
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
                if verbose:
                    im = inputs[0].detach().cpu().numpy().astype('float')
                    im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))
                    im_rec = out_generator[0].detach().cpu().numpy().astype('float')
                    im_rec = np.reshape(im_rec, (im_rec.shape[1], im_rec.shape[2], im_rec.shape[0]))
                    cv.imshow('Original Image', im)
                    cv.imshow('Reconstructed Image', im_rec)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

def evaluate(model,dataloader,threshold=0.5, device='cpu',verbose=True):
    model.eval()
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        for iteration, (inputs, labels) in enumerate(dataloader):
            out = model(inputs)
            predicted = out
            predicted[out>threshold] = 1
            predicted[out<=threshold] = 0
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Predictions/Ground truth')
            print(predicted)
            print(labels)
            if verbose:
                im = inputs[0].detach().cpu().numpy().astype('float')
                im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))

                cv.imshow('Input Image Predicted ' + str(predicted[0].cpu().detach().numpy()) + ' Label ' + str(labels[0].cpu().detach().numpy()), im)
                cv.waitKey(0)
                cv.destroyAllWindows()

        print('Accuracy of the network on the test images: %f %%' % (100*(correct/total)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, default = '../../Data/Target/')
    parser.add_argument("--test_dir", required=False, default = '../../Data/TestSet')
    parser.add_argument("--checkpoint", required=False,default = 'model.pt')

    args = parser.parse_args()

    training_dir = args.data_dir
    test_dir = args.test_dir

    image_datasets = {'train': Dataset(training_dir,mode='positive'), 'val': Dataset(test_dir,mode='mixed')}

    target_dataloader = { x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12, shuffle=True, num_workers=1) for x in ['train','val']}
    sigma = 0.155
    model = models.NoveltyDetector(noise_std=sigma**2)

    train_GAN(model.Generator, model.Discriminator,target_dataloader['train'],device='cpu',lambda_ = 0.4, discriminator_iter= 1)
    torch.save(model.state_dict(),'model.pt')
    #model.load_state_dict(torch.load('model.pt'))
    evaluate(model,target_dataloader['val'])