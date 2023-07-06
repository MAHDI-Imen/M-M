import matplotlib.pyplot as plt 

def visualize_subject(subject):
    # Visualize one subject
    print(subject)
    print(subject.image)
    print(subject.seg)
    subject.plot() #ED channel 0 #ES channel 1

def visualize_slice(image, seg):    
    # Visualize one slice
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(seg.squeeze().numpy())
    plt.title("Segmentation")

    plt.tight_layout()
    plt.show()


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