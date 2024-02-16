[**Part
4**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_]{.underline}

**What is covered:**

4.  Recognizing the need for areas of improvement in machine learning
    systems, such as job scheduling and metadata

5.  /Preventing resource starvation and avoiding deadlocks using
    scheduling techniques, such as fair-share scheduling, priority
    scheduling, and gang scheduling

6.  Handling failures more effectively to reduce any negative effect on
    users via the metadata path

-   Real-world distributed machine learning workflows are extremely
    complex, and a huge amount of *operational* work is involved to help
    maintain and manage the various components of the systems, such as
    improvements to system efficiency, observability, monitoring,
    deployment, etc.

-   In this section we'll use scheduling techniques to prevent resource
    starvation and avoid deadlocks when many team members are working
    collaboratively in the same cluster with limited computational
    resources.

-   We will also discuss the benefits of the metadata pattern, which can
    provide insights into the individual steps in machine learning
    workflows and help us handle failures more appropriately to reduce
    any negative effects on users.

#### **Operations in machine learning systems** {#operations-in-machine-learning-systems .unnumbered}

-   The focus here is on operational techniques and patterns that are
    commonly seen in more than one component or step in a machine
    learning workflow, instead of patterns that are specific to each
    individual component.

-   For example, the workflow shown in figure 6.1 includes three failed
    steps in the multiple model training steps that occur after data
    ingestion and in the multiple model serving steps that occur after
    the multiple model training steps.

-   Unfortunately, each step is like a black box, and we don't know many
    details about any of them yet.

-   At this point, we only know whether they fail and whether the
    failures have affected the following steps. As a result, they are
    really hard to debug.

> **Three steps failed in this workflow, but we don't know what the root
> cause of the failures is just by looking at the workflow at a higher
> level.**
>
> **Perhaps it failed to connect to the database or the workers for
> model training ran out of memory.**

Figure 6.1 An example workflow where multiple model training steps occur
after data ingestion and multiple model serving steps occur after the
multiple model training steps. Note the three failed steps.

-   The operation patterns introduced here can increase the visibility
    of the entire workflow to help us understand the root cause of the
    failures and give us some ideas on how to handle the failures
    properly.

```{=html}
<!-- -->
```
-   In addition, the increased observability may help us develop
    improvements in system efficiency that are beneficial to future
    executions of similar workflows.

**MLOps**

-   MLOps, a term derived from machine learning and operations, refers
    to a collection of practices for managing machine learning life
    cycles in production, including practices from machine learning and
    DevOps, to efficiently and reliably deploy and manage machine
    learning models in production.

-   MLOps requires communication and collaboration between DevOps and
    data science teams as it focuses on improving the quality of
    production machine learning and embracing automation while
    maintaining business requirements.

-   The scope of MLOps can be extremely large and varies depending on
    the context.

-   Given how large the scope of MLOps can be, depending on the context,
    we will only focus on a selected set of mature patterns at the time.

**Scheduling patterns: Assigning resources effectively in a shared
cluster**

-   A scheduler is responsible for assigning computational resources to
    perform tasks requested by the system.

-   The scheduler is designed to keep computational resources busy and
    allow multiple users to collaborate with shared resources more
    easily.

-   Multiple users are trying to build models using the shared
    computational resources in the cluster for different scenarios.

-   For example, one user is working on a fraud detection model that
    tries to identify fraudulent financial behaviors such as
    international money laundering.

-   Another user is working on a condition monitoring model that can
    generate a health score to represent the current condition for
    industrial assets such as components on trains, airplanes, wind
    turbines, etc.

-   Our beginning infrastructure only provides a simple scheduler, which
    schedules jobs on a first-come, first-served basis, as shown in
    figure 6.2. For example, the third job is scheduled after the second
    job has been scheduled, and each job's computational resources are
    allocated on scheduling.

> **The current infrastructure uses a simple scheduler that schedules
> jobs on a first-come, first-served basis.**
>
> **Job 3 is scheduled after job 2 has been scheduled.**
>
> Figure 6.2 A diagram of an infrastructure that only provides a simple
> scheduler, which schedules jobs on a first-come, first-served basis

-   When users submit multiple model training jobs to experiment with
    different sets of models or hyperparameters, these multiple models
    block prevent users' model training jobs from executing since those
    previously submitted experiments are already utilizing all the
    available computational resources.

-   Users end up competing for resources (e.g., waking up in the middle
    of the night to submit model training jobs when fewer users are
    using the system) resulting in poor collaboration among team
    members.

-   Jobs involving training very large machine learning models, which
    consume a lot of computational resources increase the time other
    users have to wait for their jobs to execute.

-   In addition, if we schedule only some of the requested workers for a
    ***distributed model training job,*** the model training ***cannot
    execute until all of the requested workers are ready***; the nature
    of the distribution strategy is distributed training with the
    collective communication pattern.

-   If necessary computational resources are lacking, the distributed
    model training job will never start, and the already-allocated
    computational resources for the existing workers will be wasted.

**The Challenge: *Find alternative approaches to first come first served
scheduling so that computational resources can be utilized much more
effectively in a shared cluster***

**The context:**

-   We have set up a distributed infrastructure for users to submit
    distributed model training jobs scheduled to run by a default
    scheduler responsible for assigning computational resources to
    perform various tasks requested by the users.

-   However, the default scheduler only provides a simple scheduler that
    schedules jobs on a first-come, first served basis.

-   As a result, when multiple users attempt to use this cluster, they
    often need to wait a long time for available computational
    resources.

-   Additionally, distributed model training jobs cannot begin to
    execute until all of the requested workers are ready due to the
    nature of the distributed training strategy, such as a collective
    communication strategy.

-   Are there any alternatives to the existing default scheduler so we
    could assign the computational resources more effectively in a
    shared cluster?

#### **The solution approach** {#the-solution-approach .unnumbered}

-   An intuitive solution approach would be to limit how much of the
    total computational resources each user is allotted.

-   If there are four users (A, B, C, and D), once user A submits a job
    that uses 25% of the total available CPU cycles
    ([https://techterms.com/definition/clockcycle)](https://techterms.com/definition/clockcycle),
    they cannot submit another job until those allocated resources are
    released and ready to be allocated to new jobs, other users could
    submit jobs independent of how much resources user A is using.

-   If user B starts two processes that use the same amount of
    resources, those processes will be attributed 12.5% of the total CPU
    cycles each, giving user B 25% of total resources

-   Each of the other users still receives 25% of the total cycles.
    Figure 6.3 illustrates the resource allocations for these four
    users.

-   Finally, if a new user E starts a process on the system, the
    scheduler will reapportion the available CPU cycles so that each
    user gets 20% of the whole (100% / 5 = 20%).

-   The way we schedule our workloads to execute in our cluster in
    figure 6.3 is called ***fair-share scheduling***.

-   It is a scheduling algorithm for computer operating systems in which
    ***the CPU usage is equally distributed among system users or
    groups, as opposed to equal distribution among processes***.

-   

> **The resources are only split among the**

**total available CPU cycles for user A. User C's resources are
independent of A.**

Figure 6.3 The resource allocations for the four users (A, B, C, and D)

-   When multiple **teams** are using the system to train their machine
    learning models and each team has multiple members, we can partition
    users into different groups and then apply the fair-share scheduling
    algorithm to ***both the users and the groups***.

-   We first divide the available CPU cycles among the groups and then
    divide further among the users within each group. For example, if
    three groups contain three, two, and four users, respectively, each
    group will be able to use 33.3% (100% / 3) of the total available
    CPU cycles. We can then calculate the available CPU cycles for each
    user in each group as follows:

```{=html}
<!-- -->
```
-   *Group 1*---33.3% / 3 users = 11.1% per user

-   *Group 2*---33.3% / 2 users = 16.7% per user

-   *Group 3*---33.3% / 4 users = 8.3% per user

```{=html}
<!-- -->
```
-   Figure 6.4 summarizes the resource allocation we calculated for each
    individual user in the three groups.

-   Fair-share scheduling would help us resolve the problem of multiple
    users running distributed training jobs concurrently.

-   We can apply this scheduling strategy at each level of abstraction,
    such as processes, users, groups, etc. All users have their own pool
    of available resources without interfering with each other.

-   However, in some situations, certain jobs should be executed
    earlier. For example, a cluster administrator would like to submit
    jobs for cluster maintenance, such as deleting jobs that have been
    stuck and taking up resources for a long time.

-   Executing these cluster maintenance jobs earlier would help make
    more computational resources available and thus unblock others from
    submitting new jobs.

Figure 6.4 A summary of the resource allocation for each user in three
groups

**Assumptions:**

-   The cluster administrator is user 1 in group 1.

-   Two other non admin users are also in group 1.

-   User 2 is running job 1, which is using all of the 11.1% of the CPU
    cycles allocated to them based on the fair-share scheduling
    algorithm.

**Details:**

-   Even though user 2 has enough computational power to perform job 1,
    the job depends on the success of job 2 from user 3. For example,
    job 2 from user 3 produces a table in the database that job 1 needs
    to perform a distributed model training task. Figure 6.5 summarizes
    the resource allocations and usages for each user in the first
    group.

-   Job 2 is stuck due to an unstable database connection and keeps
    trying to reconnect to produce the data that job 1 needs. To fix the
    problem, the administrator needs to ***submit job 3 that kills and
    then restarts the stuck job 2***.

-   Now assume that the admin user 1 is already using 11.1% of the total
    CPU cycles available.

-   Since maintenance job 3 is submitted later than all previous jobs,
    it is added to the job queue and waits to be executed when resources
    are released, based on the first-come, first-served nature of our
    fair-share scheduling algorithm.

-   As a result, we will encounter a ***deadlock*** where no job can
    proceed, as illustrated in figure 6.6.

-   To fix this problem, we can allow users to assign *priorities* to
    each of the jobs so that jobs with ***higher priority*** are
    executed earlier, ***in contrast to the first-come, first-served
    nature of the fair-share scheduling algorithm***.

Figure 6.5 A summary of resource allocations and usages for each user in
the first group

> Figure 6.6 The admin user (user 1) in group 1 is trying to schedule a
> job to restart the stuck job (job 3) but encounters a deadlock where
> no job can proceed.

-   In addition, jobs that are already running can be ***preempted or
    evicted*** to make room for jobs with higher priorities if enough
    computational resources are not available. ***This approach to
    scheduling jobs based on priorities is called priority
    scheduling.***

-   Consider, four jobs (A, B, C, and D) have been submitted
    concurrently. Each job has been marked with priorities by the users.
    Jobs A and C are **high** priority, whereas job B is **low**
    priority, and job D is **medium** priority.

-   With priority scheduling, jobs A and C will be executed first since
    they have the highest priorities, followed by the execution of job D
    with medium priority and, eventually low-priority job B. Figure 6.7
    illustrates the order of execution for the four jobs (A, B, C,
    and D) when priority scheduling is used.

> Figure 6.7 The order of execution for the four concurrently submitted
> jobs (A, B, C, and D) when priority scheduling is used

-   Consider another example. Assume three jobs (B, C, and D) with
    different priorities are submitted concurrently and are executed
    based on their priorities. If another job (job A) with high priority
    is submitted after job B, which is low priority, has already started
    running, job B will be ***preempted***, and then job A will start.

-   The computational resources previously allocated to job B will be
    released and taken over by job A. Figure 6.8 summarizes the order of
    execution for the four jobs (A, B, C, and D) where the low-priority
    job B already running is preempted by a new job (job A) with higher
    priority.

-   With priority scheduling, we can effectively ***eliminate the
    problem we previously encountered,*** where jobs can only be
    executed sequentially on a first-come, first served basis. Jobs can
    now be preempted in favor of tasks with high priorities.

-   However, f***or distributed machine learning tasks---specifically,
    model training tasks---we want to ensure that all workers are ready
    before starting distributed training***. Otherwise, the ones that
    are ready would be waiting for the remaining workers before the
    training can proceed, which wastes resources.

-   For example, in figure 6.9, three worker processes in the same
    process group are performing an allreduce operation, however, ***two
    workers are not ready because the underlying distributed cluster is
    experiencing an unstable network***.

**1. These three jobs are executed based on their priorities (C → →D
B).**

> **2. Job A (high priority) is submitted after job B (low priority) has
> already started running.**
>
> **3. Job B will be preempted, and then job A will start.**
>
> Figure 6.8 The order of execution for the four jobs (A, B, C, and D)
> where the running low-priority job is preempted by a new job with
> higher priority

-   As a result, two of the processes (processes 1 and 3) that depend on
    those affected communications ***would not receive some of the
    calculated gradient values (v0 and v2) on time (denoted by question
    marks in figure 6.9), and the entire allreduce operation is stuck
    until everything is received.***

> Figure 6.9 An example of the allreduce process with an unstable
> network between the worker processes that blocks the entire model
> training process

-   ***Gang scheduling is usually used to run distributed model training
    tasks***.

-   *It ensures that if two or more workers communicate with each other,
    they will be ready to do so at the same time; i.e., gang scheduling
    only schedules workers when enough workers are available and ready
    to communicate.*

-   If they are not gang scheduled, one worker may wait to send or
    receive a message while the other worker is sleeping, and vice
    versa. When the workers are waiting for other workers to be ready
    for communication, we are wasting allocated resources on the workers
    that are ready, and the entire distributed model training task is
    stuck.

-   For example, for collective communication--based distributed model
    training tasks, ***all workers must be ready to communicate the
    calculated gradients and update the models on each worker to
    complete an allreduce operation.***

-   Assume that the machine learning framework does not support elastic
    scheduling yet.

-   As shown in figure 6.10, the gradients are all denoted by question
    marks since they have not yet arrived in any of those worker
    processes in the second worker group. All worker processes have not
    yet started sending the gradients, and they won't until they all
    move to the ready state after the network stabilizes.

> Figure 6.10 With gang scheduling, the worker processes will not start
> sending the gradients until they are all in the ready state after the
> network becomes stable.

-   With ***gang scheduling,*** we can make sure not to start any of the
    worker processes until all workers are ready, so none of them will
    be waiting for the remaining worker processes. As a result, we can
    avoid wasting computational resources.

-   Once the network becomes stable, all of the gradients (v0, v1, and
    v2) arrive on each worker process after a successful allreduce
    operation, as shown in figure 6.11.

-   The details of different types of gang scheduling are out of the
    scope of this course
    (https://www.geeksforgeeks.org/gang-scheduling-in-operating-system/).
    We will be using an existing open source framework to integrate gang
    scheduling into distributed training in the last part of this
    course.

> Figure 6.11 All of the gradients arrive on each of the worker
> processes after a successful allreduce operation once the network is
> stable.

-   By incorporating different scheduling patterns, we are able to
    address various problems that arise when multiple users are using
    the infrastructure to schedule different types of jobs.

-   Although we looked at a few specific use cases for these scheduling
    patterns, the patterns can be found in many systems that require
    careful management of computational resources, especially when
    resources are scarce.

-   Many scheduling techniques are applied to even lower-level operating
    systems to make sure the applications run efficiently and reasonably
    share resources.

####  **Points to Note** {#points-to-note-4 .unnumbered}

-   Fair-share scheduling can help solve the problem of multiple users
    running distributed training jobs concurrently.

-   Fair-share scheduling allows the application of a scheduling
    strategy at each level of abstraction, such as processes, users,
    groups, etc.

-   Priority scheduling can be used to effectively eliminate the problem
    we encounter when jobs can only be executed sequentially on a first
    come, first-served basis.

-   Priority scheduling allows jobs to be executed based on their
    priority levels, ***preempting*** low-priority jobs to make room for
    high-priority jobs.

-   With priority scheduling, if a cluster is used by a large number of
    users, a malicious user could create jobs at the highest possible
    priority, causing other jobs to be evicted or not get scheduled at
    all.

-   To deal with this potential problem, administrators of realworld
    clusters usually enforce certain rules and limits to prevent users
    from creating a huge number of jobs at high priorities.

-   Gang scheduling ensures that, if two or more workers communicate
    with each other, they will all be ready to communicate ***at the
    same time***.

-   Gang scheduling is especially helpful for collective
    communication--based distributed model training jobs where all
    workers need to be ready to communicate the calculated gradients to
    avoid wasting computational resources.

-   Some machine learning frameworks support ***elastic scheduling***
    (https://github.com/sql-machine-learning/elasticdl/), which allows
    distributed model training jobs to start with any number of workers
    available without waiting for all the requested workers to be ready;
    If this is available, gang scheduling would **not** be preferable

-   But, because the ***number of workers may change during model
    training***, the batch size (sum of the size of mini-batches on each
    worker) will affect the model training accuracy. In that case,
    additional modifications to the model training strategy are needed.
    For example, we can support a customized learning rate scheduler
    that will account for epoch or batch or ***adjust the batch size
    dynamically*** based on the number of workers. Together with these
    algorithmic improvements, we can allocate and utilize existing
    computational resources more wisely and improve the user experience.

-   In practice, distributed model training jobs greatly benefit from
    scheduling patterns like gang scheduling. As a result, we can avoid
    wasting computational resources (which represent costs).

-   One problem we need to address is that any of these worker processes
    scheduled by gang scheduling may ***fail***, leading to unexpected
    consequences.

-   Often it's hard to debug these types of failures. and we will
    discuss patterns that will make debugging and handling failures
    easier.

#### **Quiz:** {#quiz-3 .unnumbered}

1.  Can we only apply fair-share scheduling at the user level?

2.  Is gang scheduling suitable for all distributed model training jobs?

### **Metadata pattern:**  {#metadata-pattern .unnumbered}

### **Handle failures appropriately minimize negative effects** {#handle-failures-appropriately-minimize-negative-effects .unnumbered}

-   In simple ML workflows, we can retry the failed step and easily
    continue model training without rerunning the entire data ingestion
    process, (figure 6.12).

-   However, when workflows get more complicated, failures becomes
    non-trivial to handle

-   For example, consider the workflow from previous sections.

-   That workflow trains models via three model training steps that
    arrive at different accuracies when tagging entities. Then, a model
    selection step picks the top two models with at least 90% accuracy
    trained from the first two model training steps, which will be used
    in the following two separate model serving steps.

> **Baseline workflow that includes only data ingestion, model training,
> and model serving where each of these components only appears once as
> individual steps in the workflow**
>
> **If any of the steps fail, we can easily retry the failed step and
> pick up from what's left.**
>
> Figure 6.12 A baseline workflow where the model training step has
> failed to take the ingested data. We retry the failed step and pick up
> from the failed step to continue model training without rerunning the
> entire data ingestion process.

-   The results from the two model serving steps are then aggregated via
    a result aggregation step to present to users.

-   Consider the case where the second and the third model training
    steps have both failed during execution (e.g., some of the workers
    allocated for model training are preempted). These two model
    training steps would have provided both the most and the least
    accurate model if they had finished successfully, as shown in figure
    6.13.

-   At this point, one might think that we should rerun both steps to
    proceed to the model selection and model serving steps. However, in
    practice, since we already wasted some time training part of the
    models, we may not want to start everything from scratch. It would
    be much longer before our users can see the aggregated results from
    our best models.

-   ***Question: Is there a better way to handle such kinds of
    failures?***

####  {#section-6 .unnumbered}

####  **The Challenge: To find a way to handle these failures appropriately so the negative effect on users can be minimized.** {#the-challenge-to-find-a-way-to-handle-these-failures-appropriately-so-the-negative-effect-on-users-can-be-minimized. .unnumbered}

**The context:**

-   For complicated machine learning workflows, such as the one we
    discussed in earlier sections, where we want to train multiple
    models and then select the top-performing models for model serving,
    the decision on which strategy to use to handle failures of certain
    steps due to real-world requirements is not always trivial.

-   For example, when two out of three model training steps fail due to
    preempted workers, we don't want to start training those models from
    scratch, which greatly increases the time needed to complete the
    workflow.

-   How do we handle these failures appropriately so the negative effect
    on users can be minimized?

> **Three different model training steps train**
>
> **These two model training steps would have provided both the most and
> the least accurate model if they finished successfully.**

Figure 6.13 A machine learning workflow that trains models with
different accuracies when tagging entities.

> The model selection step identifies the top two models with at least
> 90% accuracy to be used for model serving. The accuracies are crossed
> out in these two steps because the steps failed without arriving at
> the expected accuracies. The results from the two model serving steps
> are then aggregated to present to users.

#### **The solution approach:** {#the-solution-approach-1 .unnumbered}

-   Whenever we encounter a failure in a machine learning workflow, we
    should first understand the ***root cause*** (e.g., loss of network
    connections, lack of computational resources, etc).

-   Knowing the root cause is important because we need to understand
    the nature of the failure to predict whether retrying the failed
    steps would help.

-   If failures are due to long-lasting shortages, that could lead to
    ***repetitive failures*** when retrying; In this case, we could
    better utilize the computational resources to run some other tasks.

-   Figure 6.14 illustrates the difference in the effectiveness of
    retrying for permanent and temporary failures. ***Retrying the model
    training step when encountering permanent failures makes the retries
    ineffective and leads to repetitive failures.***

-   Checking whether the ***dependencies of a model training step are
    met***, viz., whether the ingested data from the previous step is
    ***still available*** is essential.

-   If the data has been ***persisted*** to a local disk to a database,
    we can proceed to model training. However, if the data was located
    in memory and lost when the model training step failed, we cannot
    start model training without ingesting the data again.

-   Figure 6.15 shows the process of restarting the data ingestion step
    when there's a permanent failure during model training.

**Permanent failures: Temporary failures:**

1.  Disappeared training data 1. Lack of resources

2.  \... 2. Loss of network connections

3.  \...

> Figure 6.14 The difference in the effectiveness of retrying for
> permanent and temporary failures
>
> **If the data was located in memory and was lost when the model
> training step failed, then we cannot start model training without
> starting ingesting the data again.**
>
> Figure 6.15 The process of restarting the data ingestion step when a
> permanent failure occurs during model training

-   Similarly, if the model training step fails due to ***preempted
    training workers or out-of-memory problems***, we need to make sure
    we still have sufficient computational resources allocated to rerun
    the model training step.

-   However, *we won't know what information to analyze **to determine
    the root cause** unless we intentionally **record it as metadata**
    during the runtime of each step in the entire machine learning
    workflow*.

-   For example, for each model training step, we can ***record
    metadata** on the availability of the ingested data and whether
    different computational resources, such as memory and CPU usage,
    exceeded the limit before the step failed*.

-   Figure 6.16 is a workflow where the model training step failed.
    Metadata is collected every 5 minutes on memory usage (in megabytes)
    and the availability of the training data (yes/no) during the
    runtime of this step.

-   We can notice a sudden huge memory spike from 23 MB to 200 MB after
    30 minutes. In this case, we can retry this step with an increase in
    requested memory, and it would then successfully produce a trained
    model that will be used for the next model serving step.

Figure 6.16 An example workflow where the model training step failed,
with the metadata collected showing an unexpected memory spike during
runtime

-   In practice, for complex workflows like in figure 6.13, even when we
    know all the dependencies of model training steps are met (e.g., we
    have enough computational resources and a good database connection
    to access the data source), we should also think about whether we
    want to handle the failures and how we'd like to handle them.

-   We've spent a lot of time on the training steps already, but now,
    the steps have suddenly failed, and we've lost all the progress. In
    other words, we don't want to start re-training all the models from
    scratch, which may add considerable time before we can deliver the
    aggregated results from our best models to users.

-   Question: ***Is there a better way to handle this without a huge
    effect on our user experience?***

-   In addition to the metadata we've recorded for each of the model
    training steps, we could save more useful metadata that can be used
    to figure out whether ***it's worth rerunning all the model training
    steps***. For example, the model accuracy over time indicates
    whether the model is being trained effectively.

-   Model accuracy that remains steady or even decreases (from 30% to
    27%, as shown in figure 6.17) may indicate that the model already
    converges and continuing training would no longer improve model
    accuracy. In this example, even though two model training steps
    fail, it's not necessary to retry the third model training step from
    scratch since it would lead to a model that converges fast but with
    low accuracy. Another example of metadata that can be potentially
    useful is the percentage of completed model training (e.g., if we've
    iterated through all the requested number of batches and epochs, the
    completion is 100%).

> **The model accuracy decreases, which might indicate that the model
> already converges and continuing training would no longer improve the
> model accuracy.**
>
> Figure 6.17 An example workflow where two model training steps fail
> and one has decreasing model accuracy

-   Once we have this additional metadata about model training steps, we
    can tell how well each started model training step progresses.

-   For example, for the workflow in figure 6.18, we could potentially
    conclude ahead of time that the third model training step was
    progressing very slowly (only 1% of completion every 30 minutes) due
    to a smaller amount of allocated computational resources or more
    complex model architecture.

-   We know that it's highly likely that, given the limited time, we end
    up with a model with low accuracy. As a result, we can disregard
    this model training step in favor of allocating more computational
    resources to the other model training steps with more potential,
    which leads to more accurate models faster.

-   Recording these metadata may help us derive more insights specific
    to each of the failed steps in the end-to-end machine learning
    workflow. We can then decide on a strategy to handle the failed
    steps appropriately to avoid wasting computational resources and
    minimize the effect on existing users.

-   The metadata patterns provide great visibility into our machine
    learning pipelines. They can also be used to search, filter, and
    analyze the artifacts produced in each step in the future if we run
    a lot of pipelines on a regular basis. For example, we might want to
    know which models are performant or which datasets contribute the
    most to those models based on the historical training metrics.

-   

> **This model training step was progressing very slowly
> ^As\ a\ result,\ we\ can^ due to smaller amount of allocated
> computational ^disregard\ this\ model^ resources or more complex model
> architecture. ^training\ step\ in\ favor^**
>
> **of allocating more computational resources to the model training
> steps with more potential, which leads to more accurate models
> faster.**
>
> Figure 6.18 An example workflow where two model training steps fail.
> One is disregarded because it is progressing very slowly, and the
> model will likely have low accuracy given the limited time.

#### **Points To Note:** {#points-to-note-5 .unnumbered}

-   With the help of the metadata pattern, we can gain additional
    insights into the individual steps in machine learning workflows.
    Then, if any fail, we can respond based on what's beneficial to our
    users and thus reduce any negative effect due to the step failures.

-   One common type of metadata is the various network performance
    ([http://mng .bz/D4lR](http://mng.bz/D4lR)) metrics while the model
    is being trained (e.g., bandwidth, throughput, latency).

-   This type of information is very useful for detecting when certain
    workers experience poor network performance that blocks the entire
    training process. We can take down slow workers and start new
    workers to continue training, assuming the underlying machine
    learning frameworks support elastic scheduling and fault-tolerance
    (see chapter 3).

-   For example, in figure 6.19, based on the metadata, the worker on
    the right-hand side has extremely high latency (10 times the latency
    of the other workers), which slows down the entire model training
    process. Ideally, this worker would be taken down and restarted.

> **This worker node has extremely high latency (10 times the latency of
> the other workers) that slows down the entire model training
> process.**
>
> Figure 6.19 An example parameter server--based model training where
> the worker on the right-hand side has extremely high latency (10 times
> the latency of the other workers), which slows down the entire model
> training process

-   One additional benefit of introducing the metadata pattern to our
    machine learning workflows is to use the metadata recorded to
    establish relationships between the individual steps or across
    different workflows. For example, modern model management tools can
    use the recorded metadata to help users build the lineage of the
    trained models and visualize what individual steps/factors
    contributed to the model artifacts.

#### **Quiz**:  {#quiz-4 .unnumbered}

1.  If the training step failed due to the loss of training data source,
    what should we do?

2.  What type of metadata can be collected if we look at individual
    workers or parameter servers?

###  {#section-7 .unnumbered}

### **Summary** {#summary-1 .unnumbered}

-   There are different areas of improvement related to operations in
    machine learning systems, such as job scheduling and metadata.

-   Various scheduling patterns, such as fair-share scheduling, priority
    scheduling, and gang scheduling, can be used to prevent resource
    starvation and avoid deadlocks.

-   We can collect metadata to gain insights from machine learning
    workflows and handle failures more appropriately to reduce any
    negative effects on users.
