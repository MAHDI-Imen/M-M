import matplotlib.pyplot as plt 
import torchvision
import numpy as np
import torch
import torchio as tio
from ..train import predict_3D


def visualize_subject(subject):
    # Visualize one subject
    print(subject)
    print(subject.image)
    print(subject.seg)
    subject.plot() #ED channel 0 #ES channel 1

def visualize_slice(image, seg, overlay=False, alpha=0.5):   
    if not overlay:
        plt.subplot(1, 2, 2)
        plt.imshow(seg.squeeze().numpy())
        plt.title("Segmentation")
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze().numpy(), cmap="gray")
        plt.title("Image")
        plt.tight_layout()


    else:
        mask = seg.squeeze().numpy()
        masked = np.ma.masked_where(mask == 0, mask)
        plt.imshow(image.squeeze().numpy(), cmap="gray")
        plt.imshow(masked,  alpha=alpha)
        plt.title("Image")
    
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

def visualize_batch(images, labels=None):
    images_grid = torchvision.utils.make_grid(images).numpy()[0]
    plt.imshow(images_grid, "gray")
    if labels is not None:
        labels_grid = torchvision.utils.make_grid(labels).numpy()[0]
        cmap = plt.colormaps.get_cmap('jet')
        mask = np.ma.masked_where(labels_grid == 0, labels_grid)
        plt.imshow(mask, cmap=cmap, alpha=0.5)
        plt.title("A batch of images")
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.show()

def visualize_predictions(images, labels, predictions, n_examples=3, step=8):
    plt.figure(figsize=(12, 4 * n_examples))

    for i in range(n_examples):
        index = i*step
        input_image = images[index]
        ground_truth = labels[index]
        predicted_mask = predictions[index]

        input_image = input_image.permute(1, 2, 0).numpy()
        ground_truth = ground_truth.permute(1, 2, 0).numpy()
        predicted_mask = predicted_mask.numpy()

        plt.subplot(n_examples, 3, i*3 + 1)
        plt.imshow(input_image, cmap='gray')
        plt.title('Input Image')

        plt.subplot(n_examples, 3, i*3 + 2)
        plt.imshow(ground_truth)
        plt.title('Ground Truth')

        plt.subplot(n_examples, 3, i*3 + 3)
        plt.imshow(predicted_mask)
        plt.title('Predicted Mask')

    plt.show()

def visualize_predictions_3D(model, subject, device):
    predictions = predict_3D(model, subject, device)

    # Restack prediction as 4D tensor 
    z = subject.image.data.shape[-1]
    predictions = torch.stack((predictions[:z],predictions[z:]))
    predictions = predictions.permute((0,2,3,1)).float()
    pred = tio.LabelMap(tensor=subject.seg.data, affine = subject.seg.affine, dtype=torch.FloatTensor)
    pred.set_data(predictions)

    new_subject = tio.Subject(
        image = subject.image,
        seg = subject.seg,
        prediction = pred
        )
    
    new_subject.plot()


# def plot_histogram(axis, tensor, num_positions=50, label=None, alpha=0.5, color=None):
#     values = tensor.numpy().ravel()
#     kernel = stats.gaussian_kde(values)
#     positions = np.linspace(values.min(), values.max(), num=num_positions)
#     histogram = kernel(positions)
#     kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
#     if label is not None:
#         kwargs['label'] = label
#     axis.plot(positions, histogram,**kwargs)


# fig, ax = plt.subplots(dpi=100)
# for subject in test_dataset:
#     tensor = subject.image.data
#     if subject.meta.Vendor=="A": color = 'red'
#     elif subject.meta.Vendor=="B": color = 'green'
#     elif subject.meta.Vendor=="C": color = 'blue'
#     else: color = 'orange'
#     plot_histogram(ax, tensor, color=color)
# ax.set_xlim(0, 0.7)
# ax.set_ylim(0, 50)
# ax.set_title('Original histograms of all samples')
# ax.set_xlabel('Intensity')
# ax.grid()


# def plot_distribution(attribute, subjects, metadata,kind='pie'):
#     data = {attribute: [metadata.loc[index][attribute] for index in subjects]}
#     partition = pd.DataFrame(data)
#     partition[attribute].value_counts().plot(kind=kind)
#     plt.title(f"Distribution over {attribute}")

# plot_distribution("Vendor", train_subjects, metadata)
#plot_distribution("Vendor", valid_subjects, metadata)


def main():
    return 0

if __name__=='__main__':
    main()