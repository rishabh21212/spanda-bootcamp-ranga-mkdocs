## **<span style="text-decoration:underline;">Section 7</span>**

**We will cover:**

-   How to handle the growing scale in large-scale machine learning
    applications

-   We will identify patterns needed to build scalable and reliable
    distributed systems

-   We will outline the use of these patterns in distributed systems and
    building reusable patterns

**Part 1 Outcome:**

-   You will be able to choose and apply the correct patterns for
    building and deploying distributed machine learning systems; use
    common tooling viz.,

    -   TensorFlow
        ([https://www.tensorflow.org)](https://www.tensorflow.org/),

    -   Kubernetes [(https://kubernetes.io)](https://kubernetes.io/),

    -   Kubeflow
        ([https://www.kubeflow.org)](https://www.kubeflow.org/),

    -   Argo Workflows

-   appropriately within a machine learning workflow; and gain practical
    experience in managing and automating machine learning tasks in
    Kubernetes.

-   The project we will explore later will help us learn to build a
    real-life distributed machine learning system using the patterns we
    study.

**ML Scale**

-   The scale of machine learning applications has become
    unprecedentedly large. Case in point: LLMs

-   Users are demanding faster responses to meet real-life requirements,
    and machine learning pipelines and model architectures are getting
    more complex.

-   Due to the growing demand and complexity, machine learning systems
    have to be built with the ability to handle the growing scale,
    including the increasing volume of historical data; frequent batches
    of incoming data; complex machine learning architectures; heavy
    model serving traffic; and complicated end-to-end machine learning
    pipelines.

**What should be done?**

-   When the dataset is too large to fit in a single machine, we need to
    store different parts of the dataset on different machines and then
    train the machine learning model by sequentially looping through the
    various parts of the dataset on different machines.

```{=html}
<!-- -->
```
-   If we have a 30 GB dataset, we can divide it into three partitions
    (fig 1.1).

Figure 1.1 Dividing a large dataset into three partitions on three
separate machines that have sufficient disk storage

-   The question to ask here is "what happens if looping through
    different parts of the dataset is quite time-consuming?".

-   Assume that the dataset at hand has been divided into three
    partitions. As in figure 1.2, we initialize the machine learning
    model on the first machine, and then we train it, using all the data
    in the first data partition. Next, we transfer the trained model to
    the second machine, which continues training by using the second
    data partition.

-   If each partition is large and time-consuming, we'll spend a
    significant amount of time waiting.

```{=html}
<!-- -->
```
-   We now need to think about adding workers and parallelism

-   Each worker is ***responsible*** for consuming each of the data
    partitions, and all workers ***train the same model in parallel***
    without waiting for others.

-   This approach is good for speeding up the model training process
    but..

    -   What if some workers finish consuming the data partitions that
        they are responsible for and want to update the model at the
        same time?

    -   Which of the worker's results (gradients) should we use to
        update the model first?

```{=html}
<!-- -->
```
-   *What about the conflicts and tradeoffs between performance and
    model quality?*

```{=html}
<!-- -->
```
-   If the data partition that the first worker uses has better quality
    due to a more rigorous data collection process than the one that the
    second worker uses, using the first worker's results first would
    produce a more accurate model.

    -   On the other hand, if the second worker has a smaller partition,
        it could finish training faster, so we could start using that
        worker's computational resources to train a new data partition.

```{=html}
<!-- -->
```
-   When more workers are added, the conflicts in completion time for
    data consumption by different workers become even more obvious.

-   Q: if the application that uses the trained machine learning model
    to make predictions observes much heavier traffic, can we simply add
    servers, with each new server handling a certain percentage of the
    traffic?

-   A: Not really. This solution would need to take other things into
    consideration, such as deciding the best load balancer strategy and
    processing duplicate requests in different servers.

-   The main takeaway for now is that we have established patterns and
    best practices to deal with certain situations, and we will use
    those patterns to make the most of our limited computational
    resources.

-   

**Distributed systems**

-   A *distributed system* is one in which components are located on
    different networked computers and can communicate with one another
    to coordinate workloads and work together via message passing.

-   Figure 1.3 below illustrates a small distributed system consisting
    of two machines communicating with each other via message passing.
    One machine contains two CPUs, and the other machine contains three
    CPUs. Obviously, a machine contains computational resources other
    than the CPUs; we use only CPUs here for illustration purposes.

-   In real-world distributed systems, the number of machines can be
    extremely large--- tens of thousands, depending on the use case.

-   Machines with more computational resources can handle larger
    workloads and share the results with other machines.

```{=html}
<!-- -->
```
-   Many patterns and reusable components are available for distributed
    systems.

-   The *work-queue pattern* in a batch processing system, for example,
    makes sure that each piece of work is independent of the others and
    can be processed without any interventions within a certain amount
    of time.

-   In addition, workers can be scaled up and down to ensure that the
    workload can be handled properly.

-   Figure 1.4 depicts seven work items, each of which might be an image
    that needs to be modified to grayscale by the system in the
    processing queue.

-   Each of the three existing workers takes two to three work items
    from the processing queue, ensuring that no worker is idle to avoid
    waste of computational resources and maximizing the performance by
    processing multiple images at the same time.

-   This performance is possible because each work item is independent
    of the others.

Figure 1.4 An example of a batch processing system using the work-queue
pattern to modify images to grayscale

**Distributed machine learning systems**

-   A *distributed machine learning system* is a distributed system
    consisting of a pipeline of steps and components that are
    responsible for different steps in machine learning applications,
    such as data ingestion, model training, and model serving.

-   It uses patterns and best practices similar to those of a
    distributed system, as well as patterns designed specifically to
    benefit machine learning applications.

-   With careful design, a distributed machine learning system is more
    scalable and reliable for handling large-scale problems, such as
    large datasets, large models, heavy model serving traffic, and
    complicated model selection or architecture optimization.

-   There are similar patterns in distributed machine learning systems.
    As an example, multiple workers can be used to train the machine
    learning model asynchronously, with each worker being responsible
    for consuming certain partitions of the dataset. This approach,
    which is similar to the work-queue pattern used in distributed
    systems, can speed up the model training process significantly.

-   Figure 1.5 illustrates how we can apply this pattern to distributed
    machine learning systems by replacing the work items with data
    partitions. Each worker takes some data partitions from the original
    data stored in a database and then uses them to train a centralized
    machine learning model.

Figure 1.5 Applying the work-queue pattern in distributed machine
learning systems

-   Another example pattern commonly used in machine learning systems
    instead of general distributed systems is the ***parameter server**
    pattern* for distributed model training.

-   As shown in figure 1.6, the parameter servers are responsible for
    storing and updating a particular part of the trained model.

-   Each worker node is responsible for taking a particular part of the
    dataset that will be used to update a certain part of the model
    parameters.

-   This pattern is useful when the model is too large to fit in a
    single server and dedicated parameter servers for storing model
    partitions without allocating unnecessary computational resources.

Figure 1.6 An example of applying the parameter server pattern in a
distributed machine learning system

**When do we use a distributed machine learning system?**

-   When any of the following scenarios occurs:

```{=html}
<!-- -->
```
-   The model is large, consisting of millions of parameters that a
    single machine cannot store and that must be partitioned on
    different machines.

-   The machine learning application needs to handle increasing amounts
    of heavy traffic that a single server can no longer manage.

-   The task at hand involves many parts of the model's life cycle, such
    as data ingestion, model serving, data/model versioning, and
    performance monitoring.

```{=html}
<!-- -->
```
-   We want to use many computing resources for acceleration, such as
    dozens of servers that have many GPUs each.

**When should we not use a distributed machine learning system?**

> If you encounter any of the following cases:

-   The dataset is small, such as a CSV file smaller than 10 GBs.

-   The model is simple and doesn't require heavy computation, such as
    linear regression.

-   Computing resources are limited but sufficient for the tasks at
    hand.

**What frameworks will we use?**

-   We'll use several popular frameworks and cutting-edge technologies
    to build components of a distributed machine learning workflow,
    including the following:

```{=html}
<!-- -->
```
-   TensorFlow
    ([https://www.tensorflow.org](https://www.tensorflow.org/))

-   Kubernetes ([https://kubernetes.io](https://kubernetes.io/))

-   Kubeflow ([https://www.kubeflow.org](https://www.kubeflow.org/))

-   Docker ([https://www.docker.com](https://www.docker.com/))

-   Argo Workflows (<https://argoproj.github.io/workflows/>)

> **Three model training**

**serving steps. to present to users.**

> Figure 1.7 End-to-end machine learning system that we will be building
>
> Table 1.1 shows the key technologies that will be covered in this
> course and example uses.

Table 1.1 The technologies covered in this course and their uses

+-------------+--------------------------------------------------------+
| Technology  | Use                                                    |
+=============+========================================================+
| TensorFlow  | Building machine learning and deep learning models     |
+-------------+--------------------------------------------------------+
| Kubernetes  | > Managing distributed environments and resources      |
+-------------+--------------------------------------------------------+
| Kubeflow    | > Submitting and managing distributed training jobs    |
|             | > easily on Kubernetes clusters                        |
+-------------+--------------------------------------------------------+
| Argo        | > Defining, orchestrating, and managing workflows      |
| Workflows   |                                                        |
+-------------+--------------------------------------------------------+
| Docker      | > Building and managing images to be used for starting |
|             | > containerized environments                           |
+-------------+--------------------------------------------------------+

https://www.linkedin.com/pulse/install-kubernetes-cluster-your-local-machine-andrea-de-rinaldis