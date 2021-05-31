from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import torch

class TSNE_project:
	def __init__(self, model, loader, args):
		images = []
		labels = []
		x_tsne = []
		model.to(args.device)
		with torch.no_grad():
			for image, label in loader:
				image = image.to(args.device)
				logits = model(image)
				x_tsne.append(logits.cpu().numpy())
				labels.append(label.numpy())
				images.append(image.cpu().numpy().reshape(image.shape[0], -1))

		x_tsne = np.array(x_tsne)
		labels = np.array(labels)
		images = np.array(images)

		x_tsne = x_tsne.reshape(-1,x_tsne.shape[2])
		labels = labels.reshape(-1)
		images = images.reshape(-1, images.shape[2])

		tsne = TSNE(n_components = 2, random_state=0)
		tsne_res = tsne.fit_transform(x_tsne)
		sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = labels, palette = sns.hls_palette(10), legend = 'full')
		plt.show()

		tsne_images = TSNE(n_components = 2, random_state=0)
		tsne_res = tsne.fit_transform(images)
		sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = labels, palette = sns.hls_palette(10), legend = 'full')
		plt.show()