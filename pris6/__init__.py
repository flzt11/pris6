import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .define_evaluation_benchmark import bench_k_means
import matplotlib.pyplot as plt
from typing import Any

# Module-level variables
reduced_data: Any = None
data: Any = None
labels: Any = None
n_digits: int = 0


def load_and_process_data():
    global data, labels, n_digits, reduced_data

    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    # Perform PCA transformation at the module level, store it in reduced_data
    reduced_data = PCA(n_components=2).fit_transform(data)


def cv(action: str):
    global data, labels, n_digits, reduced_data

    if action == "load_dataset":
        load_and_process_data()

    elif action == "run":
        if reduced_data is None:
            print("Dataset and PCA data not loaded yet. Please load the dataset first.")
            return

        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

        kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
        bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

        kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
        bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

        pca = PCA(n_components=n_digits).fit(data)
        kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
        bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

        print(82 * "_")

    elif action == "visualize":
        if reduced_data is None:
            print("Dataset and PCA data not loaded yet. Please load the dataset first.")
            return

        kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.title(
            "K-means clustering on the digits dataset (PCA-reduced data)\n"
            "Centroids are marked with white cross"
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    else:
        print(f"Action '{action}' not recognized.")
