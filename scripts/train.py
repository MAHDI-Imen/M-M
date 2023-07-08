from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def save_model(model_name, model, optimizer, num_epochs, total_steps, train_losses, valid_losses):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': {
            'num_epochs': num_epochs,
            'total_steps': total_steps,
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
    }, f'models/{model_name}.pth')


def prepare_data(images, labels, device):
    images = images.to(device)
    labels = labels.squeeze().long().to(device)
    labels_onehot = F.one_hot(labels, num_classes=4).permute(0, 3, 1, 2).float().to(device)
    return images, labels_onehot


def train_epoch(model, optimizer, criterion, train_loader, device):
    epoch_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels_onehot = prepare_data(images, labels, device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return model, epoch_loss / len(train_loader)


def valid_epoch(model, criterion,valid_dataset, device):
    model.eval()
    images, labels = valid_dataset[:][0], valid_dataset[:][1]
    images, labels_onehot =  prepare_data(images, labels, device)

    with torch.no_grad():
        valid_outputs = model(images)
        valid_loss = criterion(valid_outputs, labels_onehot)
    
    return(valid_loss.item())



def train_model(model, 
               optimizer, 
               criterion, 
               device,  
               train_dataset, 
               valid_dataset=None,
               batch_size=64,
               num_workers=2,
               num_epochs=100,
               verbose=0,
               save=True,
               model_name=None,
               ):
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers
                              )
    total_steps = len(train_loader)   


    if verbose==0:
        epochs = range(num_epochs)
    else:
        epochs = tqdm(range(num_epochs))


    model.to(device)
    train_losses = []
    valid_losses = []
    for epoch in epochs:
        model, epoch_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        train_losses.append(epoch_loss)

        if valid_dataset is not None:
            valid_loss = valid_epoch(model, criterion, valid_dataset, device)
            valid_losses.append(valid_loss)

        if verbose>1:
            if epoch % 10 == 0:
                if valid_dataset is not None:
                    print(f"Epoch [{epoch+1}/{num_epochs}]], Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{num_epochs}]], Loss: {train_losses[-1]:.4f}")


    if save:
        save_model(model_name, model, optimizer, num_epochs, total_steps, train_losses, valid_losses)

    return model


def predict(model, images, device):
    images = images.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(images)

    probabilities = torch.softmax(output, dim=1)
    _, predictions = torch.max(probabilities, dim=1)
    predictions = predictions.cpu()
    return predictions


def predict_3D(model, subject, device):
    image = subject.image.data
    c, x, y, z = image.shape

    # Stack 4D image as a batch of 2D images
    stacked = image.permute((0,3,1,2)).reshape(c*z, 1 ,x, y)
    predictions = predict(model, stacked, device)

    return predictions



def main():
    return 0

if __name__=='__main__':
    main()