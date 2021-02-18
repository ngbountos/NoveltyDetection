import argparse
import torch
from torch import optim, nn
import numpy as np
from Dataset import Dataset
import models
import cv2 as cv


def add_noise(std, batch, device='cpu'):
    x = batch
    noise = np.random.normal(0, std, size=x.shape)
    noise = torch.from_numpy(noise).float()
    x_noisy = (x + noise).clamp(0.0, 1.0)

    return x.to(device), x_noisy.to(device)


def train_GAN(generator, discriminator, dataloader, device='cpu', lambda_=0.4, discriminator_iter=1, generator_iter=1,
              std=0.155 ** 2, verbose=True):
    generator.train()
    discriminator.train()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0, 0.9))
    optimizerG = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    criterion = nn.BCELoss()
    criterion_gen = nn.MSELoss()
    discriminator_iterations = discriminator_iter
    generator_iterations = generator_iter
    loss_mse = 1000
    epoch = 0
    while loss_mse > 0.01 or epoch < 20:
        flagD = True
        flagG = False
        counterD = 0
        counterG = 0

        for iteration, (inputs, labels) in enumerate(dataloader):

            # Fix Generator and train discriminator
            if flagD:
                flagG = False
                print('In discriminator epoch %d iteration %d Counter %d' % (epoch, iteration, counterD))
                # update discriminator
                optimizerD.zero_grad()
                # Add noise for robustness
                x, x_noisy = add_noise(std, inputs, device)
                out_generator = generator(x_noisy)
                # Calculate loss of discriminator on generated input
                discr_generator = discriminator(out_generator)
                discr_generator = discr_generator.view((discr_generator.shape[0]))
                zeros = torch.zeros(len(discr_generator), dtype=torch.float, device=device)

                loss_discr_generated = criterion(discr_generator, zeros)
                # Calculate loss of discriminator on real input
                discr_original = discriminator(x)
                discr_original = discr_original.view((discr_original.shape[0]))

                ones = torch.ones(len(discr_original), dtype=torch.float, device=device)

                loss_discr_original = criterion(discr_original, ones)
                # Sum discriminator loss
                loss_discr = loss_discr_generated + loss_discr_original
                # backpropagation
                loss_discr.backward()
                optimizerD.step()
                if iteration % 1 == 0:
                    print('Total Discriminator Loss %f Iteration %d ' % (loss_discr, iteration))
                counterD += 1
                if counterD >= discriminator_iterations:
                    flagG = True
                    counterD = 0

            if flagG:
                flagD = False
                # Fix Discriminator and Train Generator
                optimizerG.zero_grad()
                print('In generator epoch %d iteration %d Counter %d' % (epoch, iteration, counterG))

                # Add noise for robustness
                x, x_noisy = add_noise(std, inputs, device)
                out_generator = generator(x_noisy)
                # Get discriminator's decision
                discrimator_decision_generated = discriminator(out_generator)
                discrimator_decision_generated = discrimator_decision_generated.view(
                    (discrimator_decision_generated.shape[0]))

                ones = torch.ones(len(discrimator_decision_generated), dtype=torch.float, device=device)
                # Loss from min-max game
                loss_generator = criterion(discrimator_decision_generated, ones)
                # Loss from Encoder-Decoder module
                reconstruction_loss = criterion_gen(out_generator, x)
                # Loss Combination
                generator_loss = loss_generator + lambda_ * reconstruction_loss
                # generator_loss = reconstruction_loss
                # Backpropagtion
                generator_loss.backward()
                optimizerG.step()
                print('Total Generator Loss %f Iteration %d ' % (generator_loss, iteration))
                loss_mse = reconstruction_loss.item()
                print('MSE Loss %f' % loss_mse)
                if iteration % 9 == 0 and epoch % 5 == 0:
                    if verbose:
                        im = inputs[0].detach().cpu().numpy().astype('float')
                        im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))
                        im_rec = out_generator[0].detach().cpu().numpy().astype('float')
                        im_rec = np.reshape(im_rec, (im_rec.shape[1], im_rec.shape[2], im_rec.shape[0]))
                        cv.imshow('Original Image', im)
                        cv.imshow('Reconstructed Image', im_rec)
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                counterG += 1
                if counterG >= generator_iterations:
                    flagD = True
                    counterG = 0
        epoch += 1


def evaluate(model, dataloader, threshold=0.5, device='cpu', verbose=True):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        total = 0.0
        correct = 0.0
        correct_true = 0.0
        total_true = 0.0
        for iteration, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            predicted = out

            predicted[out > threshold] = 1
            predicted[out <= threshold] = 0
            predicted = predicted.squeeze()
            total += inputs.shape[0]
            t = predicted == labels
            correct += (predicted == labels).sum().item()
            correct_true += (predicted[labels==1]==labels[labels==1]).sum().item()
            total_true += labels[labels==1].shape[0]
            print(predicted)
            print(labels)
            if verbose:
                im = inputs[0].detach().cpu().numpy().astype('float')
                im = np.reshape(im, (im.shape[1], im.shape[2], im.shape[0]))

                cv.imshow('Input Image Predicted ' + str(predicted[0].cpu().detach().numpy()) + ' Label ' + str(
                    labels[0].cpu().detach().numpy()), im)
                cv.waitKey(0)
                cv.destroyAllWindows()

        print('Accuracy of the network on the test images: %f %%' % (100 * (correct / total)))
        print('True Positive Accuracy of the network on the test images: %f %%' % (100 * (correct_true / total_true)))


def train_classifier(net, trainloader, testloader, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    best_val = 0
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:
                print('Epoch %d, Iteration %5d loss: %.5f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
                print('Accuracy of the network on the training set: %d %%' % (
                        100 * correct / total))

        correct = 0
        total = 0
        correct_true = 0
        total_true = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_true += (predicted[labels==1]==labels[labels==1]).sum().item()
                total_true += labels[labels==1].shape[0]

        print('Accuracy of the network on the val images: %d %%' % (
                100 * correct / total))
        print(' True positive Accuracy of the network on the val images: %d %%' % (
                100 * correct_true / total_true))
        if correct_true/total_true > best_val:
            print('New best val acc / Saving model')
            torch.save(net.state_dict(), 'modelCl.pt')
            best_val = correct_true/total_true

    print('Finished Training')


def start_classification_training(training_dir, val_dir, test_dir):
    image_datasets = {'train': Dataset(training_dir, mode='mixed'), 'val': Dataset(val_dir, mode='mixed', set='val'), 'test': Dataset(test_dir, mode='mixed', set='val')}
    target_dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=1)
                         for x in ['train', 'val','test']}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torchvision
    model =  torchvision.models.densenet121() #models.Classifier()
    model.classifier = nn.Linear(1024,2)
    model = model.to(device)

    model = model.to(device)
    train_classifier(model, target_dataloader['train'], target_dataloader['val'],device=device)
    model.load_state_dict(torch.load('modelCl.pt'))
    model.eval()
    correct = 0
    total = 0
    correct_true = 0
    total_true = 0
    with torch.no_grad():
            for data in target_dataloader['test']:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_true += (predicted[labels==1]==labels[labels==1]).sum().item()
                total_true += labels[labels==1].shape[0]

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            print(' True positive Accuracy of the network on the test images: %d %%' % (
                    100 * correct_true / total_true))


    #torch.save(model.state_dict(), 'modelCl.pt')


def start_GAN_training(training_dir, test_dir):
    image_datasets = {'train': Dataset(training_dir, mode='positive'),
                      'val': Dataset(test_dir, mode='mixed', set='val')}

    target_dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12, shuffle=True, num_workers=1)
                         for x in ['train', 'val']}
    sigma = 0.155

    model = models.NoveltyDetector(noise_std=sigma ** 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_GAN(model.Generator, model.Discriminator, target_dataloader['train'], device=device, lambda_=0.4,
              discriminator_iter=1, generator_iter=1, verbose=False)
    torch.save(model.state_dict(), 'model.pt')
    # model.load_state_dict(torch.load('model.pt'))
    evaluate(model, target_dataloader['val'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False, default='../Data/Dataset')
    parser.add_argument("--test_dir", required=False, default='../Data/TestSet')
    parser.add_argument("--checkpoint", required=False, default='model.pt')

    args = parser.parse_args()

    training_dir = args.data_dir
    test_dir = args.test_dir

    start_classification_training(training_dir, test_dir)

    #start_GAN_training(training_dir,test_dir)
