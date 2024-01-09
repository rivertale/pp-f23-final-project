# pp-f23-final-project

## Title
Accelerating K-Means Clustering with Parallelization

## Member
311554014 陳冠昇 
312551114 朱立民 
312551096 方嘉賢 

## How to run the code
**Serial**
```c=1
cd sequential/
make
cd build/
./kmeans input_image_path output_image_path
```

**Pthread**


**OpenMP**
```c=1
cd parallel_openmp/
make
./openMP input_image_path output_image_path
```

**Cuda**
```c=1
cd parallel_cuda/
make
./KMeans input_image_path output_image_path
```

**Some of option**

    -n number of clusters (default is 4)

    -m max iteration of kmean clustering (default is 200)

    -r exit when the data point migration ratio between clusters exceeds this value (default is 0.01)

    -q quiet mode (no output)

    -h print this help information

## Abstract
In this project, our goal is to enhance the computational speed of the image K-Means clustering algorithm through parallelization methods. By adopting three different parallelization approaches, namely Pthread, OpenMP, and CUDA, we have successfully achieved a significantly improved computational efficiency for the K-Means clustering algorithm compared to the serial version. The experimental results clearly indicate a substantial speed boost in the CUDA version when handling substantial computations. On the other hand, Pthread and OpenMP, while showing comparable performance improvements, both outperform the serial version.

Key Words: K-Means, Clustering, Machine Learning , Pthread, OpenMP, Cuda

## Itroduction
K-Means clustering is a crucial method in the fields of machine learning and data analysis. It effectively groups data into similar clusters, revealing patterns and structures within the data. This clustering method is not only helpful in uncovering underlying data structures and trends but also finds extensive applications in areas such as image segmentation, anomaly detection, and data compression. Through K-Means clustering, we gain deeper insights into the similarities and differences within a dataset, providing a foundation for subsequent analysis and decision-making.

The motivation behind this project lies in optimizing the computational efficiency of K-Means clustering through parallel computing methods to enhance processing speed. As the volume of data continues to increase, traditional K-Means algorithms can become computationally intensive. Therefore, accelerating K-Means execution through parallelization aids in improving analysis efficiency and addressing challenges posed by larger datasets. This optimization holds significant value for real-time applications, resource maximization, and faster cluster analysis.

The core idea of K-Means clustering involves assigning data points to K cluster centers, minimizing the distance between each data point and its corresponding cluster center. This grouping ensures that data points within each cluster are similar, while points across different clusters exhibit larger differences. Optimizing the efficiency of this process is key to enhancing the performance of K-Means clustering.

Through parallel computing, we can simultaneously process multiple data points, accelerating the computation of K-Means. This parallelization method is well-suited for modern computing environments, utilizing multi-core processors or distributed computing systems to achieve faster cluster analysis. This not only aids in handling large-scale datasets but also enhances the algorithm's scalability, making it applicable to various application scenarios.

The project goals include developing a parallelized implementation of the K-Means algorithm and optimizing the computation steps for data clustering. This will contribute to reducing processing time, making K-Means clustering more suitable for applications requiring real-time analysis or handling big data.

Ultimately, the outcomes of this project will have a positive impact across various domains. This optimization not only increases the speed and efficiency of analysis but also provides a better foundation for more in-depth data exploration and pattern discovery. Therefore, optimizing K-Means clustering through parallel computing is a forward-looking and practically valuable research endeavor, playing a crucial role in advancing the fields of machine learning and data analysis.

## Proposed solution
**Serial**

Our serial approach divides the K-Means clustering algorithm into 4 steps, Figure below shows the flowchart of the serial approach.

<img  width="256" src="https://i.imgur.com/hoG9ufI.png" title="source: imgur.com" />

1. Initialize cluster centroids
Select K initial cluster centroids.To ensure consistency, we use the first K distinct pixel values in the image as the initial centroids for K-Means.
2. Classify points
Assign each pixel value to the nearest cluster center, forming K clusters. When calculating distances, employ the Euclidean distance.
3. Update centroids
Update the centroid of each cluster by averaging the pixels within the same cluster.
4. Check cluster switching proportion
To avoid excessive execution times, the loop is exited when the overall pixel change falls below our predefined threshold indicating convergence.


**Pthread**

Figure below gives an overview of our Pthread implementation and shows how it parallelizes the steps in the serial approach. The main parts being parallelized are the three states: classifying points, updating centroids, and checking cluster switching proportion.

<img width="256" src="https://i.imgur.com/Xzwjv20.png" title="source: imgur.com" />

In each iteration, the image is divided equally among threads. Each thread is responsible for classifying a subset of the image and calculating local centroids based on the pixels it processes. As the local centroids are computed by averaging the pixels within the same cluster, we can subsequently average these local centroids to derive the global centroids to use in the next iteration by using the pixel count in local centroids as the weight.

It is crucial to note that local centroids are updated frequently with every pixel. Therefore, to prevent any false sharing, it is important to separate the local centroid data on different cache lines. To address this we pad the local centroids to 128 bytes to avoid any potential sharing of cache lines between different threads.

**OpenMP**

In the implementation of OpenMP parallelization, we parallelize three key parts of the algorithm. 

Firstly, in the "Classify points" section, we need to calculate the shortest Euclidean distance between all pixel values and cluster centers, and record to which cluster each pixel belongs. As each pixel's calculation is independent, we can utilize **#pragma omp** for to achieve parallelization. 

Next, in the "Update centroids" section, we iterate through each pixel using a for loop to accumulate numerical values for updating cluster centers. To ensure correctness, we include **reduction** in **#pragma omp** for to handle variable accumulation. 

Finally, in the "Check cluster switching proportion" section, we similarly use **#pragma omp**  for for parallelization and apply **reduction** to handle variable accumulation. 

Through these modifications, we have achieved OpenMP parallelization in all three sections, enhancing the computational efficiency of the K-Means clustering algorithm, particularly suitable for handling large-scale pixel data scenarios.

**Cuda**

CUDA parallelization will be performed for two main tasks: classifying points and updating centroids. 

Firstly, to avoid frequent loading of large amounts of image data into the GPU, we combine these two tasks into a single kernel function. 

For the classification of points, each pixel reads cluster centroids to calculate distances, eliminating the possibility of race conditions.
However, updating centroids involves counting the number of points in each cluster, introducing the potential for race conditions. To address this, I declare the counting variable as **\_\_shared\_\_**, making it private to each thread block, preventing multiple threads from simultaneously accessing the same space. Additionally, I use **atomicAdd()** to make the count write operation atomic, ensuring that only one thread writes to this variable at a time. Finally, **\_\_syncthreads()** is employed to synchronize actions at each stage.

## Experimental Methodolog
**Experimental platform**

Our experiments will be conducted using the workstation provided in class. Table below outlines the specifications of the workstation, including both hardware and software details.

|||
| ------------- | ------ |
|OS|Ubuntu 20.04|
|CPU|i5|
|Cores|4|
|Treads|4|
|GPU|GTX 1060 6 GB|
|Cuda version|10.1|

**Input image**

The input image for our experiments is Lena, as depicted in Figure below. Throughout the experiments, we will vary the image sizes to 64X64, 128X128, 256X256, and 512X512, conducting different experiments for each size.

<img  width="256" src="https://i.imgur.com/pXd3T0n.png" title="source: imgur.com" />

## Experimental Results
**Different thread count**

It can be seen from the figure that there is no linear relationship.

<img  width="512" src="https://i.imgur.com/RBLvTJ7.png" title="source: imgur.com" />

**Different image size**

When the number of clusters is set to 32

<img width="512" src="https://i.imgur.com/LaAtzUG.png" title="source: imgur.com" />

When the number of clusters is set to 4

<img  width="512" src="https://i.imgur.com/WyIjov3.png" title="source: imgur.com" />

**Different cluster count**

The experiment is conducted using an image size of 512X512 and is tested with 4 clusters

<img width="512" src="https://i.imgur.com/umma69K.png" title="source: imgur.com" />

**Output Result**

<img width="512" src="https://i.imgur.com/YxZTJMF.png" title="source: imgur.com" />

## Related Work
K-Means Clustering is a simple algorithm that reveals the relationship between data. K-Means has two major challenges. First, in order to converge to a set of centroids, the algorithm must traverse each pixel in every iteration to update the centroids. Second, the initial selection of centroids will largely affect the total number of iterations. 

Previous work has tried to improve the traversing efficiency by analyzing the data using model selection[2] and the hierarchical structure of k-d tree[3]. These techniques  demonstrate notable speedup on sequential architectures, but the algorithm they used are not suitable for parallelization due to its branching behavior.

As for the initial selection, the main idea in [1] is to select multiple set of initial centroids and pick the best set to use as the initial centroids within few iterations, since this approach evaluate many initial sets at the same time, it can be parallelized easily.

## Conclusions

In our experiments with serial, OpenMP, pthread, and CUDA implementations, similar to time complexity, we can observe that the size of the image and the number of clusters are correlated to the required computation.

Among the three different parallelization methods, CUDA stands out as the most performant method, especially in scenarios involving substantial computations. While Pthread and OpenMP demonstrated similar results in experiments, Pthread showed slight advantages in certain situations.

The performance impact of image size varied across parallelization methods, with CUDA consistently outperforming Pthread and OpenMP for larger images. However, for smaller images, the need for CUDA to move the image across GPU memory and CPU memory brings a significant amount of overhead, resulting in worse performance compared to Pthread and OpenMP. 

In summary, our work enhances the performance of serial K-Means clustering algorithm through parallelization with OpenMP, Pthread, and CUDA. CUDA can achieve higher performance on larger images, while Pthread and OpenMP perform better on very small images. By analyzing the image size and cluster count in use, our work provides valuable insights for selecting suitable parallelization methods and thread count and ultimately leads to better performance in image processing and data analysis.

## References

[1]Janki Bhimani, Miriam Leeser, and Ningfang Mi. 2015. Accelerating K-Means
clustering with parallel implementations and GPU computing. In 2015 IEEE High
Performance Extreme Computing Conference (HPEC). 1–6. 
https://doi.org/10.1109/HPEC.2015.7322467

[2] T. Kanungo, D.M. Mount, N.S. Netanyahu, C.D. Piatko, R. Silverman, and A.Y. Wu.
2002. An efficient k-means clustering algorithm: analysis and implementation.
IEEE Transactions on Pattern Analysis and Machine Intelligence 24, 7 (2002), 881–892.
https://doi.org/10.1109/TPAMI.2002.1017616

[3] Dan Pelleg and Andrew W. Moore. 2000. X-Means: Extending K-Means with
Efficient Estimation of the Number of Clusters. In Proceedings of the Seventeenth
International Conference on Machine Learning (ICML ’00). Morgan Kaufmann
Publishers Inc., San Francisco, CA, USA, 727–734.

## Contributions of each member

311554014 陳冠昇: Serial、OpenMP實作、投影片製作

312551114 朱立民: Cuda 實作、投影片製作

312551096 方嘉賢: pthread實作、投影片製作

