## **<span style="text-decoration:underline;">Section 15</span>**

**FL Server - Python Implementation**
-   The server-side implementation of a **federated learning** (**FL**)
    system is critical for realizing authentic FL-enabled applications.

-   We have discussed the basic system architecture and flow in the
    previous section. Here, more hands-on implementation will be
    discussed so that you can create a simple server and aggregator of
    the FL system that various **machine learning** (**ML**)
    applications can be connected to and tested on.

-   This section describes an actual implementation aspect of FL
    server-side components discussed in *section 3*.

-   Based on the understanding of how the entire process of the FL
    system works, you will be able to go one step further to make it
    happen with example code provided here and on GitHub.

-   Once you understand the basic implementation principles using the
    example code, it is a fun aspect to be able enhance the FL server
    functionalities based on your own design.

-   Here, we're going to cover the following topics:

```{=html}
<!-- -->
```
-   Main software components of the aggregator

-   Implementing FL server-side functionalities

-   Maintaining models for aggregation with the state manager

-   Aggregating local models

-   Running the FL server

-   Implementing and running the database server

-   Potential enhancements to the FL server

Technical requirements

-   All the code files introduced Here can be found on GitHub
    here: [[https://github.com/keshavaspanda/simple-fl]{.underline}](https://github.com/keshavaspanda/simple-fl)

Main software components of the aggregator  and database

-   The architecture of an aggregator with the FL server was introduced
    in the previous section. Here, we will introduce the code that
    realizes the basic functionalities of an FL system.

-   The aggregator and database-side Python-based software components
    are listed in the aggregator directory of fl_main, as well
    as lib/util and pseudodb folders, as in *Figure 4.1*:

![Figure 4.1 -- Python software components for the aggregator as well as
internal libraries and pseudo database
](.\\images\\/media/image19.jpg){width="3.407120516185477in"
height="3.9518897637795276in"}

Figure 4.1 -- Python software components for the aggregator as well as
internal libraries and pseudo database

-   The following is a brief description of the Python code files in the
    aggregator.

The Aggregator-side code

-   In this section, we will touch on the main Python files of the
    aggregator-side related to the FL server thread, FL state manager,
    and model aggregation itself.

-   These aggregator-side code files are found in the aggregator folder.
    The code in the repo only captures the model aggregation
    perspective, not the entire engineering aspects of creating a
    thorough FL platform.

FL server code (server_th.py)

-   This is the main code that realizes the whole basic flow of the FL
    process from the communication processes between an aggregator
    itself, agents, and a database to coordinating agent participation
    and the aggregation of the ML models.

-   It also initializes the global cluster model sent from the first
    connected agent. It manages receiving local models and the cluster
    model synthesis routine in which the cluster global model is formed
    after collecting enough local models.

FL state manager (state_manager.py)

-   The state manager buffers the local model and cluster model data
    that is needed for aggregation processes.

-   The buffers will be filled out when the aggregator receives local
    models from the agents and cleared when proceeding to the next round
    of the FL process.

-   The checking function of the aggregation criteria is also defined in
    this file.

Aggregation code (aggregation.py)

-   The aggregation Python code will list the basic algorithms for
    aggregating the model.

-   In the code example used here Here, we will only introduce the
    averaging method called **federated averaging** (**FedAvg**),
    which averages the weights of the collected local models considering
    local dataset sizes to generate a cluster global model.

lib/util codes

-   The Python files for the internal libraries
    (communication_handler.py, data_struc.py, helpers.py, messengers.py,
    and states.py) will be explained in the *Appendix*, *Exploring
    Internal Libraries*.

Database-side code

-   Database-side code consists of the pseudo database and the SQLite
    database Python code files that can be found in the pseudodb folder.

-   The pseudo database code is hosting a server to receive messages
    from the aggregator and parse them to process as the ML model data
    that can be utilized for the FL process.

Pseudo database code (pseudo_db.py)

-   The function of pseudo database Python code is to accept the
    messages related to the local and global cluster models from the
    aggregator and push the information to the database. It also saves
    the ML model binary files in the local file system.

SQLite database code (sqlite_db.py)

-   The SQLite database Python code creates an actual SQLite database at
    the specified path. It also has the function to insert data entries
    related to the local and global cluster models into the database.

-   Now that the aggregator and database-side software components are
    defined, let\'s move on to the configuration of the aggregator.

Configuring the aggregator

-   The following code is an example of the aggregator-side
    configuration parameters defined in the config_aggregator.json file,
    which can be found in the setups folder:

> {
>
>     \"aggr_ip\": \"localhost\",
>
>     \"db_ip\": \"localhost\",
>
>     \"reg_socket\": \"8765\",
>
>     \"exch_socket\": \"7890\",
>
>     \"recv_socket\": \"4321\",
>
>     \"db_socket\": \"9017\",
>
>     \"round_interval\": 5,
>
>     \"aggregation_threshold\": 1.0,
>
>     \"polling\": 1
>
> }

CopyExplain

-   The parameters include the aggregator's IP (the FL server's IP), the
    database server's IP, and the various port numbers of the database
    and agents.

-   The round interval is the time of the interval at which the criteria
    of aggregation are checked and the aggregation threshold defines the
    percentage of collected local ML models needed to start the
    aggregation process. The polling flag is related to whether to
    utilize the polling method for communications between the aggregator
    and agents or not.

-   Now that we have covered the concept of the configuration file for
    the aggregator side, let's move on to how the code is designed and
    implemented.

FL server-side functions

-   In this section, we will explain how you can implement the very
    first version of an aggregator with an FL server system using the
    actual code examples, which are in server_th.py in
    the aggregator directory.

-   This way, you will understand the core functionalities of the FL
    server system and how they are implemented so that you can further
    enhance a lot more functionalities on your own.

-   Therefore, we will only cover the important and core functionalities
    that are critical to conducting a simple FL process. The potential
    enhancements will be listed in the later section of This
    section, *Potential enhancements to the FL server*.

-   server_th.py handles all the aspects of basic functionalities
    related to the FL server side, so let's look into that in the
    following section.

FL Server Library Imports

-   The FL server-side code starts with importing the necessary
    libraries. In particular, lib.util handles the basic supporting
    functionalities to make the implementation of FL easy. The details
    of the code can be found in the GitHub repository.

-   The server code imports StateManager and Aggregator for the FL
    processes. The code about the state manager and aggregation will be
    discussed in later sections Here about *Maintaining models for
    aggregation with the state manager *and* Aggregating local models.*

-   Here is the code for importing the necessary libraries:

> import asyncio, logging, time, numpy as np
>
> from typing import List, Dict, Any
>
> from fl_main.lib.util.communication_handler import init_fl_server,
> send, send_websocket, receive
>
> from fl_main.lib.util.data_struc import convert_LDict_to_Dict
>
> from fl_main.lib.util.helpers import read_config, set_config_file
>
> from fl_main.lib.util.messengers import generate_db_push_message,
> generate_ack_message, generate_cluster_model_dist_message,
> generate_agent_participation_confirmation_message
>
> from fl_main.lib.util.states import ParticipateMSGLocation,
> ModelUpMSGLocation, PollingMSGLocation, ModelType, AgentMsgType
>
> from .state_manager import StateManager
>
> from .aggregation import Aggregator

CopyExplain

-   After we import the necessary libraries, let us move on to designing
    an FL Server class.

Defining the FL Server class

-   In practice, it is wise to define the Server class, using which you
    can create an instance of the FL server that has the functionalities
    discussed in *earlier*, as follows:

> class Server:
>
>     \"\"\"
>
>     FL Server class defining the functionalities of
>
>     agent registration, global model synthesis, and
>
>     handling mechanisms of messages by agents.
>
>     \"\"\"

CopyExplain

-   Again, the server class primarily provides the functionalities of
    agent registration and global model synthesis and handles the
    mechanisms of uploaded local models and polling messages sent from
    agents. It also serves as the interface between the aggregator and
    database and between the aggregator and agents.

-   The FL server class functionality is now clear -- next is
    initializing and configuring the server.

Initializing the FL server

-   The following code inside the \_\_init\_\_ constructor is an example
    of the initialization process of the Server instance:

> def \_\_init\_\_(self):
>
>     config_file = set_config_file(\"aggregator\")
>
>     self.config = read_config(config_file)
>
>     self.sm = StateManager()
>
>     self.agg = Aggregator(self.sm)
>
>     self.aggr_ip = self.config\[\'aggr_ip\'\]
>
>     self.reg_socket = self.config\[\'reg_socket\'\]
>
>     self.recv_socket = self.config\[\'recv_socket\'\]
>
>     self.exch_socket = self.config\[\'exch_socket\'\]
>
>     self.db_ip = self.config\[\'db_ip\'\]
>
>     self.db_socket = self.config\[\'db_socket\'\]
>
>     self.round_interval = self.config\[\'round_interval\'\]
>
>     self.is_polling = bool(self.config\[\'polling\'\])
>
>     self.sm.agg_threshold =
>
>                      self.config\[\'aggregation_threshold\'\]

CopyExplain

-   Then, self.config stores the information from
    the config_aggregator.json file discussed in the preceding code
    block.

-   self.sm and self.agg have instances of the state manager class and
    aggregator class discussed as follows, respectively.

-   self.aggr_ip reads an IP address from the aggregator's configuration
    file.

-   Then, reg_socket and recv_socket will be set up, where reg_socket is
    used for agents to register themselves together with an aggregator
    IP address stored as self.aggr_ip, and recv_socket is used for
    receiving local models from agents, together with an aggregator IP
    address stored as self.aggr_ip. Both reg_socket and recv_socket in
    this example code can be read from the aggregator's configuration
    file.

-   The exch_socket is the port number used to send the global model
    back to the agent together with the agent IP address, which is
    initialized with the configuration parameter in the initialization
    process.

-   The information to get connected to the database server will then be
    configured, where dp_ip and db_socket will be the IP address and the
    port number of the database server, respectively, all read from
    the config_aggregator.json file.

-   round_interval is an interval time to check whether the aggregation
    criteria for starting the model aggregation process are met or not.

-   The is_polling flag is related to whether to use the polling method
    from the agents or not. The polling flag must be the same as the one
    used in the agent-side configuration file.

-   agg_threshold is also the percentage over the number of collected
    local models that is used in
    the ready_for_local_aggregation function where if the percentage of
    the collected models is equal to or more than agg_threshold, the FL
    server starts the aggregation process of the local models.

-   Both self.round_interval and self.agg_threshold are read from the
    configuration file in this example code too.

-   Now that the configuration has been set up, we will talk about how
    to register agents that are trying to participate in the FL process.

Agent Registration

-   In this section, the simplified and asynchronous register function
    is described to receive the participation message specifying the
    model structures and return socket information for future model
    exchanges. It also sends the welcome message back to the agent as a
    response.

-   The registration process of agents is described in the following
    example code:

> async def register(self, websocket: str, path):        
>
>     msg = await receive(websocket)
>
>     es = self.\_get_exch_socket(msg)
>
>     agent_nm = msg\[int(ParticipateMSGLocation.agent_name)\]
>
>     agent_id = msg\[int(ParticipateMSGLocation.agent_id)\]
>
>     ip = msg\[int(ParticipateMSGLocation.agent_ip)\]
>
>     id, es = self.sm.add_agent(agent_nm, agent_id, ip, es)
>
>     if self.sm.round == 0:
>
>         await self.\_initialize_fl(msg)
>
>     await self.\_send_updated_global_model( \\
>
>         websocket, id, es)

CopyExplain

-   In this example code, the received message from an agent, defined
    here as msg, is decoded by the receive function imported from
    the communication_handler code.

-   In particular, the self.sm.add_agent(agent_name, agent_id, addr,
    es) function takes the agent name, agent ID, agent IP address, and
    the exch_socket number included in the msg message in order to
    accept the messages from this agent, even if the agent is
    temporarily disconnected and then connected again.

-   After that, the registration function checks whether it should move
    on to the process of initial models or not, depending on the FL
    round that is tracked with self.sm.round. If the FL process is not
    happening yet, that is, if self.sm.round is 0, it calls
    the \_initialize_fl(msg) function in order to initialize the FL
    process.

-   Then, the FL server sends the updated global model back to the agent
    by calling the \_send_updated_global_model(websocket, id,
    es) function. The function takes the WebSocket, agent ID,
    and exch_socket as parameters and creates a reply message to the
    agent to notify it whether the participation message has been
    accepted or not.

-   The registration process of agents with the FL server is simplified
    in this example code here. In a production environment, all the
    system information from the agent will be pushed to the database so
    that an agent that loses the connection to the FL server can be
    recovered anytime by reconnecting to the FL server.

-   Usually, if the FL server is installed in the cloud and agents are
    connected to the FL server from their local environment, this
    push-back mechanism from the aggregator to agents will not work
    because of security settings such as firewalls.

-   We do not discuss the topic of security issues in detail, so you are
    encouraged to use the polling method implemented in
    the simple-fl code to communicate between the cloud-based aggregator
    and local agents.

Getting socket information to push the global model back to agents

-   The following function called \_get_exch_socket takes
    a participation message from the agent and decides which port to use
    to reach out to the agent depending on the simulation flag in the
    message:

> def \_get_exch_socket(self, msg):
>
>     if msg\[int(ParticipateMSGLocation.sim_flag)\]:
>
>         es = msg\[int(ParticipateMSGLocation.exch_socket)\]
>
>     else:
>
>         es = self.exch_socket
>
>     return es

CopyExplain

-   We support a simulation run in this implementation exercise by which
    you can run all the FL system components of a database, aggregator,
    and multiple agents in one machine.

-   Initializing the FL process if necessary

-   The asynchronous \_initialize_fl function is for initializing an FL
    process that is only called when the round of FL is 0. The following
    is the code to do so:

> async def \_initialize_fl(self, msg):
>
>     agent_id = msg\[int(ParticipateMSGLocation.agent_id)\]
>
>     model_id = msg\[int(ParticipateMSGLocation.model_id)\]
>
>     gene_time = msg\[int(ParticipateMSGLocation.gene_time)\]
>
>     lmodels = msg\[int(ParticipateMSGLocation.lmodels)\]
>
>     perf_val = msg\[int(ParticipateMSGLocation.meta_data)\]
>
>     init_flag = \\
>
>         bool(msg\[int(ParticipateMSGLocation.init_flag)\])
>
>     self.sm.initialize_model_info(lmodels, init_flag)
>
>     await self.\_push_local_models( \\
>
>         agent_id, model_id, lmodels, gene_time, perf_val)
>
>     self.sm.increment_round()

CopyExplain

-   After extracting the agent ID (agent_id), the model ID (model_id),
    local models from an agent (lmodels), the generated time of the
    model (gene_time), the performance data (perf_val), and the value
    of init_flag from the received message,
    the initialize_model_info function of the state manager code is
    called, which is explained in a later section of This section.

-   This function then pushes the local model to the database by calling
    the \_push_local_models function, which is also described in this
    section. You can refer to the *Functions to push the local and
    global models to the database* section.

-   After that, the round is incremented to proceed to the first round
    in FL.

Confirming agent participation with an updated global model

-   After initializing the (cluster) global model, the global models
    need to be sent to the agent connected to the aggregator through
    this registration process. The
    asynchronous \_send_updated_global_model function as follows handles
    the process of sending the global models to the agent by taking the
    WebSocket information, agent ID, and the port to use to reach out to
    the agent as parameters. The following code block describes the
    procedure:

> async def \_send_updated_global_model( \\
>
>                    self, websocket, agent_id, exch_socket):
>
>     model_id = self.sm.cluster_model_ids\[-1\]
>
>     cluster_models = \\
>
>        convert_LDict_to_Dict(self.sm.cluster_models)
>
>     reply = generate_agent_participation_confirm_message(
>
>        self.sm.id, model_id, cluster_models, self.sm.round,
>
>        agent_id, exch_socket, self.recv_socket)
>
>     await send_websocket(reply, websocket)

CopyExplain

-   If the FL process has already started, that is, the self.sm.round is
    more than 0 already, we get the cluster models from their buffer and
    convert them into a dictionary format with
    the convert_LDict_to_Dict library function.

-   Then, the reply message is packaged using
    the generate\_ agent_participation_confirm_message function and sent
    to the agent that just connected or reconnected to the aggregator by
    calling the send_websocket(reply, websocket) function. Please also
    refer to the *Functions to send the global models to the
    agents* section.

-   Now that we understand the agents' registration process, let's move
    on to the implementation of handling the local ML models and polling
    messages.

The server for handling messages from local agents

-   The asynchronous receive_msg_from_agent process at the FL server is
    constantly running to receive local model updates and to push them
    to the database and the memory buffer temporally saving local
    models. It also responds to the polling messages from the local
    agents. The following code explains this functionality:

> async def receive_msg_from_agent(self, websocket, path):
>
> msg = await receive(websocket)
>
>     if msg\[int(ModelUpMSGLocation.msg_type)\] == \\
>
>                                        AgentMsgType.update:
>
>         await self.\_process_lmodel_upload(msg)
>
>     elif msg\[int(PollingMSGLocation.msg_type)\] == \\
>
>                                       AgentMsgType.polling:
>
>         await self.\_process_polling(msg, websocket)  

CopyExplain

-   We will then look into the two functions called by
    the receive_msg_from_agent function as shown in the preceding code
    blocks, which are
    the \_process_lmodel_upload and \_process_polling functions.

Processing a model upload by local agents

-   The asynchronous \_process_lmodel_upload function deals with
    the AgentMsgType.update message. The following code block is about
    the function related to receiving the local ML models and putting
    them into the buffer in the state manager:

> async def \_process_lmodel_upload(self, msg):
>
>     lmodels = msg\[int(ModelUpMSGLocation.lmodels)\]
>
>     agent_id = msg\[int(ModelUpMSGLocation.agent_id)\]
>
>     model_id = msg\[int(ModelUpMSGLocation.model_id)\]
>
>     gene_time = msg\[int(ModelUpMSGLocation.gene_time)\]
>
>     perf_val = msg\[int(ModelUpMSGLocation.meta_data)\]
>
>     await self.\_push_local_models( \\
>
>         agent_id, model_id, lmodels, gene_time, perf_val)
>
>     self.sm.buffer_local_models( \\
>
>         lmodels, participate=False, meta_data=perf_val)

CopyExplain

-   First, it extracts the agent ID (agent_id), the model ID (model_id),
    local models from an agent (lmodels), the generated time of the
    model (gene_time), and the performance data (perf_val) from the
    received message, and then calls the \_push_local_models function to
    push the local models to the database.

-   The buffer_local_models function is then called to save the local
    models (lmodels) in the memory buffer.
    The buffer_local_models function is described in the *Maintaining
    models for aggregation with the state manager* section.

Processing polling by agents

-   The following asynchronous \_process_polling function deals with
    the AgentMsgType.polling message:

> async def \_process_polling(self, msg, websocket):
>
>     if self.sm.round \> \\
>
>                    int(msg\[int(PollingMSGLocation.round)\]):
>
>         model_id = self.sm.cluster_model_ids\[-1\]
>
>         cluster_models = \\
>
>             convert_LDict_to_Dict(self.sm.cluster_models)
>
>         msg = generate_cluster_model_dist_message( \\
>
>             self.sm.id, model_id, self.sm.round, \\
>
>             cluster_models)
>
>         await send_websocket(msg, websocket)
>
>     else:
>
>         msg = generate_ack_message()
>
>         await send_websocket(msg, websocket)  

CopyExplain

-   If the FL round (self.sm.round) is greater than the local FL round
    included in the received message that is maintained by the local
    agent itself, it means that the model aggregation is done during the
    period between the time when the agent polled to the aggregator last
    time and now.

-   In this case, cluster_models that are converted into a dictionary
    format are packaged into a response message
    by generate_cluster_model_dist_message and sent back to the agent
    via the send_websocket function.

-   Otherwise, the aggregator just returns the *ACK* message to the
    agent, generated by the generate_ack_message function.

-   Now we are ready to aggregate the local models received from the
    agents, so let us look into the model aggregation routine.

The global model synthesis routine

-   The global model synthesis routine process designed in async def
    model_synthesis_routine(self) in the FL server periodically checks
    the number of stored models and executes global model synthesis if
    there are enough local models collected to meet the aggregation
    threshold.

-   The following code describes the model synthesis routine process
    that periodically checks the aggregation criteria and executes model
    synthesis:

> async def model_synthesis_routine(self):
>
>     while True:
>
>         await asyncio.sleep(self.round_interval)
>
>         if self.sm.ready_for_local_aggregation():  
>
>             self.agg.aggregate_local_models()
>
>             await self.\_push_cluster_models()
>
>             if self.is_polling == False:
>
>                 await self.\_send_cluster_models_to_all()
>
>             self.sm.increment_round()

CopyExplain

-   This process is asynchronous, running with a while loop.

-   In particular, once the criteria set
    by ready_for_local_aggregation (explained in the *Maintaining models
    for aggregation with the state manager* section) are met,
    the aggregate_local_models function imported from
    the aggregator.py file is called, where this function averages the
    weights of the collected local models based on FedAvg. Further
    explanation of the aggregate_local_models function can be found in
    the *Aggregating local models* section.

-   Then, await self.\_push_cluster_models() is called to push the
    aggregated cluster global model to the database.

-   await self.\_send_cluster_models_to_all() is for sending the updated
    global model to all the agents connected to the aggregator if
    the polling method is not used.

-   Last but not least, the FL round is incremented
    by self.sm.increment_round().

-   Once the cluster global model is generated, the models need to be
    sent to the connected agents with the functions described in the
    following section.

Functions to send the global models to the agents

-   The functionality of sending global models to the connected agents
    is dealt with by the \_send_cluster_models_to_all function. This is
    an asynchronous function to send out cluster global models to all
    agents under this aggregator as follows:

> async def \_send_cluster_models_to_all(self):
>
>     model_id = self.sm.cluster_model_ids\[-1\]
>
>     cluster_models = \\
>
>         convert_LDict_to_Dict(self.sm.cluster_models)
>
>     msg = generate_cluster_model_dist_message( \\
>
>         self.sm.id, model_id, self.sm.round, \\
>
>         cluster_models)
>
>     for agent in self.sm.agent_set:
>
>         await send(msg, agent\[\'agent_ip\'\], agent\[\'socket\'\])

CopyExplain

-   After getting the cluster models' information, it creates the
    message including the cluster models, round, model ID, and
    aggregator ID information using
    the generate_cluster_model_dist_message function and calls
    the send function from the communication_handler libraries to send
    the global models to all the agents in the agent_set registered
    through the agent participation process.

-   Sending the cluster global models to the connected agents has now
    been explained. Next, we explain how to push the local and cluster
    models to the database.

Functions to push the local and global models to the database

-   The \_push_local_models and \_push_cluster_models functions are both
    called internally to push and send the local models and cluster
    global models to the database.

Pushing local models to the database

-   Here is the \_push_local_models function for pushing a given set of
    local models to the database:

> async def \_push_local_models(self, agent_id: str, \\
>
>         model_id: str, local_models: Dict\[str, np.array\], \\
>
>         gene_time: float, performance: Dict\[str, float\]) \\
>
>         -\> List\[Any\]:
>
>     return await self.\_push_models(
>
>         agent_id, ModelType.local, local_models, \\
>
>         model_id, gene_time, performance)

CopyExplain

-   The \_push_local_models function takes parameters such as the agent
    ID, local models, the model ID, the generated time of the model, and
    the performance data, and returns a response message if there is
    one.

-   Pushing cluster models to the database

-   The following \_push_cluster_models function is for pushing the
    cluster global models to the database:

> async def \_push_cluster_models(self) -\> List\[Any\]:
>
>     model_id = self.sm.cluster_model_ids\[-1\]
>
>     models = convert_LDict_to_Dict(self.sm.cluster_models)
>
>     meta_dict = dict({ \\
>
>         \"num_samples\" : self.sm.own_cluster_num_samples})
>
>     return await self.\_push_models( \\
>
>         self.sm.id, ModelType.cluster, models, model_id, \\
>
>         time.time(), meta_dict)

CopyExplain

-   \_push_cluster_models in this code does not take any parameters, as
    those parameters can be obtained from the instance information and
    buffered memory data of the state manager.

-   For example, self.sm.cluster_model_ids\[-1\] obtains the ID of the
    latest cluster model, and self.sm.cluster_models stores the latest
    cluster model itself, which is converted into models with a
    dictionary format to be sent to the database. It also
    creates mata_dict to store the number of samples.

-   Pushing ML models to the database

-   Both the preceding functions call the \_push_models function as
    follows:

> async def \_push_models(
>
>     self, component_id: str, model_type: ModelType,
>
>     models: Dict\[str, np.array\], model_id: str,
>
>     gene_time: float, performance_dict: Dict\[str, float\])
>
>     -\> List\[Any\]:
>
>     msg = generate_db_push_message(component_id, \\
>
>         self.sm.round, model_type, models, model_id, \\
>
>         gene_time, performance_dict)
>
>     resp = await send(msg, self.db_ip, self.db_socket)
>
>     return resp

CopyExplain

-   In this code example, the \_push_models function takes parameters
    such as component_id (the ID of the aggregator or
    agent), model_type, such as local or cluster
    model, models themselves, model_id, gene_time (the time the model is
    created), and performance_dict as the performance metrics of the
    models.

-   Then, the message to be sent to the database (using
    the send function) is created by
    the generate_db_push_message function, taking these parameters
    together with the FL round information. It returns a response
    message from the database.

-   Now that we have explained all the core functionalities related to
    the FL server, let us look into the role of the state manager, which
    maintains all the models needed for the aggregation process.

Maintaining models for aggregation with the  state manager

-   In this section, we will explain state_manager.py, which handles
    maintaining the models and necessary volatile information related to
    the aggregation of local models.

State Manager Library Imports

-   This code imports the following. The internal libraries
    for data_struc, helpers, and states are introduced in
    the *Appendix*, *Exploring Internal Libraries*:

> import numpy as np
>
> import logging
>
> import time
>
> from typing import Dict, Any
>
> from fl_main.lib.util.data_struc import LimitedDict
>
> from fl_main.lib.util.helpers import generate_id, generate_model_id
>
> from fl_main.lib.util.states import IDPrefix

CopyExplain

-   After importing the necessary libraries, let's define the state
    manager class.

Defining the state manager class

-   The state manager class (Class StateManager), as seen
    in state_manager.py, is defined in the following code:

> class StateManager:
>
>     \"\"\"
>
>     StateManager instance keeps the state of an aggregator.
>
>     Functions are listed with this indentation.
>
>     \"\"\"

CopyExplain

-   This keeps track of the state information of an aggregator. The
    volatile state of an aggregator and agents should also be stored,
    such as local models, agents' info connected to the aggregator,
    cluster models generated by the aggregation process, and the current
    round number.

-   After defining the state manager, let us move on to initializing the
    state manager.

Initializing the state manager

-   In the \_\_init\_\_ constructor, the information related to the FL
    process is configured. The following code is an example of how to
    construct the state manager:

> def \_\_init\_\_(self):
>
>     self.id = generate_id()
>
>     self.agent_set = list()
>
>     self.mnames = list()
>
>     self.round = 0
>
>     self.local_model_buffers = LimitedDict(self.mnames)
>
>     self.local_model_num_samples = list()
>
>     self.cluster_models = LimitedDict(self.mnames)
>
>     self.cluster_model_ids = list()
>
>     self.initialized = False
>
>     self.agg_threshold = 1.0

CopyExplain

-   The ID of the self.id aggregator can be generated randomly using
    the generate_id() function from the util.helpers library.

-   self.agent_set is a set of agents connected to the aggregator where
    the format of the set is a collection of dictionary information,
    related to agents in this case.

-   self.mnames stores the names of each layer of the ML models to be
    aggregated in a list format.

-   self.round is initialized to be 0 so that the round of FL is
    initialized.

-   local_model_buffers is a list of local models collected by agents
    stored in the memory space. local_model_buffers accepts the local
    models sent from the agents for each FL round, and once the round is
    completed by the aggregation process, this buffer is cleared and
    starts accepting the next round's local models.

-   self.local_model_num_samples is a list that stores the number of
    data samples for the models that are collected in the buffer.

-   self.cluster_models is a collection of global cluster models in
    the LimitedDict format, and self.cluster_model_ids is a list of IDs
    of cluster models.

-   self.initialized becomes True once the initial global model is set
    and is False otherwise.

-   self.agg_threshold is initialized to be 1.0, which is overwritten by
    the value specified in the config_aggregator.json file.

-   After initializing the state manager, let us investigate
    initializing a global model next.

Initializing a global model

-   The following initialize_model_info function sets up the initial
    global model to be used by the other agents:

> def initialize_model_info(self, lmodels, \\
>
>                           init_weights_flag):
>
>     for key in lmodels.keys():
>
>         self.mnames.append(key)
>
>     self.local_model_buffers = LimitedDict(self.mnames)
>
>     self.cluster_models = LimitedDict(self.mnames)
>
>     self.clear_lmodel_buffers()
>
>     if init_weights_flag:
>
>         self.initialize_models(lmodels, \\
>
>                             weight_keep=init_weights_flag)
>
>     else:
>
>         self.initialize_models(lmodels, weight_keep=False)

CopyExplain

-   It fills up the model names (self.mnames) extracted from the local
    models (lmodels) sent from an initial agent.

-   Together with the model
    names, local_model_buffers and cluster_models are re-initialized
    too. After clearing the local model buffers, it calls
    the initialize_models function.

-   The following initialize_models function initializes the structure
    of neural networks (numpy.array) based on the initial base models
    received as parameters of models with a dictionary format
    (str or np.array):

> def initialize_models(self, models: Dict\[str, np.array\], \\
>
>                                 weight_keep: bool = False):
>
>     self.clear_saved_models()
>
>     for mname in self.mnames:
>
>         if weight_keep:
>
>             m = models\[mname\]
>
>         else:
>
>             m = np.zeros_like(models\[mname\])
>
>         self.cluster_models\[mname\].append(m)
>
>         id = generate_model_id(IDPrefix.aggregator, \\
>
>                  self.id, time.time())
>
>         self.cluster_model_ids.append(id)
>
>         self.initialized = True

CopyExplain

-   For each layer of the model, defined here as model names, this
    function fills out the model parameters. Depending on
    the weight_keep flag, the model is initialized with zeros or
    parameters that are received.

-   This way, the initial cluster global model is constructed together
    with the randomized model ID. If an agent sends a different ML model
    than the model architecture defined here, the aggregator rejects the
    acceptance of the model or gives an error message to the agent.
    Nothing is returned.

-   So, we have covered initializing the global model. In the following
    section, we will explain the core part of the FL process, which is
    checking aggregation criteria.

Checking the aggregation criteria

-   The following code, called ready_for_local_aggregation, is
    for checking the aggregation criteria:

> def ready_for_local_aggregation(self) -\> bool:
>
>     if len(self.mnames) == 0:
>
>             return False
>
>     num_agents = int(self.agg_threshold \* \\
>
>                                        len(self.agent_set))
>
>     if num_agents == 0: num_agents = 1
>
>     num_collected_lmodels = \\
>
>         len(self.local_model_buffers\[self.mnames\[0\]\])
>
>     if num_collected_lmodels \>= num_agents:
>
>         return True
>
>     else:
>
>         return False            

CopyExplain

-   This ready_for_local_aggregation function returns a bool value to
    identify whether the aggregator can start the aggregation process.
    It returns True if it satisfies the aggregation criteria (such as
    collecting enough local models to aggregate) and False otherwise.
    The aggregation threshold, agg_threshold, is
    configured in the config_aggregator.json file.

-   The following section is about buffering the local models that are
    used for the aggregation process.

Buffering the local models

-   The following code on buffer_local_models stores local models from
    an agent in the local model buffer:

> def buffer_local_models(self, models: Dict\[str, np.array\],
>
>         participate=False, meta_data: Dict\[Any, Any\] = {}):
>
>     if not participate:  
>
>         for key, model in models.items():
>
>             self.local_model_buffers\[key\].append(model)
>
>         try:
>
>             num_samples = meta_data\[\"num_samples\"\]
>
>         except:
>
>             num_samples = 1
>
>         self.local_model_num_samples.append( \\
>
>                 int(num_samples))
>
>     else:  
>
>         pass
>
>     if not self.initialized:
>
>         self.initialize_models(models)

CopyExplain

-   The parameters include the local models formatted as a dictionary as
    well as meta-information such as the number of samples.

-   First, this function checks whether the local model sent from an
    agent is either the initial model or not by checking the
    participation flag. If it is an initial model, it calls
    the initialize_model function, as shown in the preceding code block.

-   Otherwise, for each layer of the model defined with model names, it
    stores the numpy array in the self.local_model_buffers. The key is
    the model name and model mentioned in the preceding code are the
    actual parameters of the model.

-   Optionally, it can accept the number of samples or data sources that
    the agent has used for the retraining process and push it to
    the self. local_model_num_samples buffer.

-   This function is called when the FL server receives the local models
    from an agent during the receive_msg_from_agent routine.

-   With that, the local model buffer has been explained. Next, we will
    explain how to clear the saved models so that aggregation can
    continue without having to store unnecessary models in the buffer.

Clearing the saved models

-   The following clear_saved_models function clears all cluster models
    stored in this round:

> def clear_saved_models(self):
>
>     for mname in self.mnames:
>
>         self.cluster_models\[mname\].clear()

CopyExplain

-   This function is called when initializing the FL process at the very
    beginning and the cluster global model is emptied to start a fresh
    FL round again.

-   The following function, the clear_lmodel_buffers function, clears
    all the buffered local models to prepare for the next FL round:

> def clear_lmodel_buffers(self):
>
>     for mname in self.mnames:
>
>         self.local_model_buffers\[mname\].clear()
>
>     self.local_model_num_samples = list()

CopyExplain

-   Clearing the local models in local_model_buffers is critical when
    proceeding to the next FL round. Without this process, the models to
    be aggregated are mixed up with the non-relevant models from other
    rounds, and eventually, the performance of the FL is sometimes
    degraded.

-   Next, we will explain the basic framework of adding agents during
    the FL process.

Adding agents

-   This add_agent function deals with brief agent registration using
    system memory:

> def add_agent(self, agent_name: str, agent_id: str, \\
>
>                                agent_ip: str, socket: str):
>
>     for agent in self.agent_set:
>
>         if agent_name == agent\[\'agent_name\'\]:
>
>             return agent\[\'agent_id\'\], agent\[\'socket\'\]
>
>     agent = {
>
>         \'agent_name\': agent_name,
>
>         \'agent_id\': agent_id,
>
>         \'agent_ip\': agent_ip,
>
>         \'socket\': socket
>
>     }
>
>     self.agent_set.append(agent)
>
>     return agent_id, socket

CopyExplain

-   This function just adds agent-related information to
    the self.agent_set list. The agent information includes the agent
    name, agent ID, agent IP address, and the socket number to reach out
    to the agent.

-   The socket number can be used when sending the cluster global model
    to the agent connected to the aggregator and when the push method is
    used for communication between an aggregator and an agent.

-   This function is only called during the agent registration process
    and returns the agent ID and the socket number.

-   If the agent is already registered, which means there is already an
    agent with the same name in agent_set, it returns the agent ID and
    the socket number of the existing agent.

-   Again, this push communication method from an aggregator to agents
    does not work under certain security circumstances. It is
    recommended to use the polling method that the agents use to
    constantly check whether the aggregator has an updated global model
    or not.

-   The agent registration mechanism can be expanded using a database,
    which will give you better management of the distributed systems.

-   Next, we will touch on incrementing the FL round.

Incrementing the FL round

-   The increment_round function just increments the round number
    precisely managed by the state manager:

> def increment_round(self):
>
>     self.round += 1

CopyExplain

-   Incrementing rounds is a critical part of the FL process for
    supporting the continuous learning operation. This function is only
    called after registering the initial global model or after each
    model aggregation process.

-   Now that we understand how the FL works with the state manager, in
    the following section, we will talk about the model aggregation
    framework.

Aggregating local models

-   The aggregation.py code handles aggregating local models with a
    bunch of aggregation algorithms. In the code example, we only
    support **FedAvg**, as discussed in the following sections.

Importing the libraries for the aggregator

-   The aggregation.py code imports the following:

> import logging
>
> import time
>
> import numpy as np
>
> from typing import List
>
> from .state_manager import StateManager
>
> from fl_main.lib.util.helpers import generate_model_id
>
> from fl_main.lib.util.states import IDPrefix

CopyExplain

-   The imported state manager's role and functionalities are discussed
    in the *Maintaining models for aggregation with the state
    manager* section, and the helpers and states libraries are
    introduced in the *Appendix*, *Exploring Internal Libraries.*

\- After importing the necessary libraries, let's define the aggregator
class.

Defining and initializing the aggregator class

-   The following code for class Aggregator defines the core process of
    the aggregator, which provides a set of mathematical functions for
    computing the aggregated models:

> class Aggregator:
>
>     \"\"\"
>
>     Aggregator class instance provides a set of
>
>     mathematical functions to compute aggregated models.
>
>     \"\"\"

CopyExplain

-   The following \_\_init\_\_ function just sets up the state manager
    of the aggregator to access the model buffers:

> def \_\_init\_\_(self, sm: StateManager):
>
>     self.sm = sm

CopyExplain

-   Once the aggregator class is defined and initialized, let's look at
    the actual FedAvg algorithm implementation.

Defining the aggregate_local_models function

-   The following aggregate_local_models function is the code for
    aggregating the local models:

> def aggregate_local_models(self):
>
>     for mname in self.sm.mnames:
>
>         self.sm.cluster_models\[mname\]\[0\] \\
>
>             = self.\_average_aggregate( \\
>
>                 self.sm.local_model_buffers\[mname\], \\
>
>                 self.sm.local_model_num_samples)
>
>     self.sm.own_cluster_num_samples = \\
>
>         sum(self.sm.local_model_num_samples)
>
>     id = generate_model_id( \\
>
>         IDPrefix.aggregator, self.sm.id, time.time())
>
>     self.sm.cluster_model_ids.append(id)
>
>     self.sm.clear_lmodel_buffers()

CopyExplain

-   This function can be called after the aggregation criteria are
    satisfied, such as the aggregation threshold defined in
    the config_aggregator.json file. The aggregation process uses local
    ML models buffered in the memory of the state manager.

-   Those local ML models are sent from the registered agents. For each
    layer of the models defined by mname, the weights of the model are
    averaged by the \_average_aggregate function as follows to realize
    FedAvg. After averaging the model parameters of all the
    layers, cluster_models is updated, which is sent to all the agents.

-   Then, the local model buffer is cleared to be ready for the next
    round of the FL process.

The FedAvg function

-   The following function, \_average_aggregate, called by the
    preceding aggregate_local_models function, is the code that realizes
    the FedAvg aggregation method:

> def \_average_aggregate(self, buffer: List\[np.array\],
>
>                        num_samples: List\[int\]) -\> np.array:
>
>     denominator = sum(num_samples)
>
>     model = float(num_samples\[0\])/denominator \* buffer\[0\]
>
>     for i in range(1, len(buffer)):
>
>         model += float(num_samples\[i\]) /
>
>                                     denominator \* buffer\[i\]
>
>     return model

CopyExplain

-   In the \_average_aggregate function, the computation is simple
    enough that, for each buffer of the given list of ML models, it
    takes averaged parameters for the models.

-   The basics of model aggregation are discussed earlier. It returns
    the weighted aggregated models with np.array.

-   Now that we have covered all the essential functionalities of the FL
    server and aggregator, next, we will talk about how to run the FL
    server itself.

Running the FL server

-   Here is an example of running the FL server. In order to run the FL
    server, you will just execute the following code:

> if \_\_name\_\_ == \"\_\_main\_\_\":
>
>     s = Server()
>
>     init_fl_server(s.register,
>
>                    s.receive_msg_from_agent,
>
>                    s.model_synthesis_routine(),
>
>                    s.aggr_ip, s.reg_socket, s.recv_socket)

CopyExplain

-   The register, receive_msg_from_agnet,
    and model_synthesis_routine functions of the instance of the FL
    server are for starting the registration process of the agents,
    receiving messages from the agents, and starting the model synthesis
    process to create a global model, which are all started using
    the init_fl_server function from
    the communication_handler libraries.

-   We have covered all the core modules of the aggregator with the FL
    server. They can work with the database server, which will be
    discussed in the following section.

Implementing and running the database server

-   The database server can be hosted either on the same machine as the
    aggregator server or separately from the aggregator server.

-   Whether the database server is hosted on the same machine or not,
    the code introduced here is still applicable to both cases.

-   The database-related code is found in the fl_main/pseudodb folder of
    the GitHub repository provided alongside this book.

Configuring the database

-   The following code is an example of the database-side configuration
    parameters saved as config_db.json:

> {
>
>     \"db_ip\": \"localhost\",
>
>     \"db_socket\": \"9017\",
>
>     \"db_name\": \"sample_data\",
>
>     \"db_data_path\": \"./db\",
>
>     \"db_model_path\": \"./db/models\"
>
> }

CopyExplain

-   In particular, db_data_path is the location of the SQLite database
    and db_model_path is the location of the ML model binary files.

-   The config_db.json file can be found in the setup folder.

-   Next, let's define the database server and import the necessary
    libraries.

Defining the database server

-   The main functionality of the pseudo_db.py code is accepting
    messages that contain local and cluster global models.

-   Importing the libraries for the pseudo database

-   First, the pseudo_db.py code imports the following:

> import pickle, logging, time, os
>
> from typing import Any, List
>
> from .sqlite_db import SQLiteDBHandler
>
> from fl_main.lib.util.helpers import generate_id, read_config,
> set_config_file
>
> from fl_main.lib.util.states import DBMsgType, DBPushMsgLocation,
> ModelType
>
> from fl_main.lib.util.communication_handler import init_db_server,
> send_websocket, receive

CopyExplain

-   It imports the basic general libraries as well
    as SQLiteDBHandler (discussed later in the *Defining the database
    with SQLite *section) and the functions from the lib/util libraries
    that are discussed in the *Appendix*, *Exploring Internal
    Libraries*.

Defining the PseudoDB class

-   The PseudoDB class is then defined to create an instance that
    receives models and their data from an aggregator and pushes them to
    an actual database (SQLite, in this case):

> class PseudoDB:
>
>     \"\"\"
>
>     PseudoDB class instance receives models and their data
>
>     from an aggregator, and pushes them to database
>
>     \"\"\"

CopyExplain

-   Now, let us move on to initializing the instance of PseudoDB.

Initializing PseudoDB

-   Then, the initialization process, \_\_init\_\_, is defined as
    follows:

> def \_\_init\_\_(self):
>
>     self.id = generate_id()
>
>     self.config = read_config(set_config_file(\"db\"))
>
>     self.db_ip = self.config\[\'db_ip\'\]
>
>     self.db_socket = self.config\[\'db_socket\'\]
>
>     self.data_path = self.config\[\'db_data_path\'\]
>
>     if not os.path.exists(self.data_path):
>
>         os.makedirs(self.data_path)
>
>     self.db_file = \\
>
>         f\'{self.data_path}/model_data{time.time()}.db\'
>
>     self.dbhandler = SQLiteDBHandler(self.db_file)
>
>     self.dbhandler.initialize_DB()
>
>     self.db_model_path = self.config\[\'db_model_path\'\]
>
>     if not os.path.exists(self.db_model_path):
>
>         os.makedirs(self.db_model_path)

CopyExplain

-   The initialization process generates the ID of the instance and sets
    up various parameters such as the database socket (db_socket), the
    database IP address (db_ip), the path to the database (data_path),
    and the database file (db_file), all configured from config_db.json.

-   dbhandler stores the instance of SQLiteDBHandler and calls
    the initialize_DB function to create an SQLite database.

-   Folders for data_path and db_model_path are created if they do not
    already exist.

-   After the initialization process of PseudoDB, we need to design the
    communication module that accepts the messages from the aggregators.

-   We again use WebSocket for communicating with an aggregator and
    start this module as a server to accept and respond to messages from
    an aggregator.

-   In this design, we do not push messages from the database server to
    an aggregator or agents in order to make the FL mechanism simpler.

-   Handling messages from the aggregator

-   The following code for the async def handler function,
    which takes websocket as a parameter, receives messages from the
    aggregator and returns the requested information:

> async def handler(self, websocket, path):
>
>     msg = await receive(websocket)
>
>     msg_type = msg\[DBPushMsgLocation.msg_type\]
>
>     reply = list()
>
>     if msg_type == DBMsgType.push:
>
>         self.\_push_all_data_to_db(msg)
>
>         reply.append(\'confirmation\')
>
>     else:
>
>         raise TypeError(f\'Undefined DB Message Type: \\
>
>                                               {msg_type}.\')
>
>     await send_websocket(reply, websocket)

CopyExplain

-   In the handler function, once it decodes the received message from
    an aggregator, the handler function checks whether the message type
    is push or not.

-   If so, it tries to push the local or cluster models to the database
    by calling the \_push_all_data_to_db function.

-   Otherwise, it will show an error message. The confirmation message
    about pushing the models to the database can then be sent back to
    the aggregator.

-   Here, we only defined the type of the push message, but you can
    define as many types as possible, together with the enhancement of
    the database schema and design.

Pushing all the data to the database

-   The following code for \_push_all_data_to_db pushes the models'
    information to the database:

> def \_push_all_data_to_db(self, msg: List\[Any\]):
>
>     pm = self.\_parse_message(msg)
>
>     self.dbhandler.insert_an_entry(\*pm)
>
>     model_id = msg\[int(DBPushMsgLocation.model_id)\]
>
>     models = msg\[int(DBPushMsgLocation.models)\]
>
>     fname = f\'{self.db_model_path}/{model_id}.binaryfile\'
>
>     with open(fname, \'wb\') as f:
>
>         pickle.dump(models, f)

CopyExplain

-   The models' information is extracted by the \_parse_message function
    and passed to the \_insert_an_entry function.

-   Then, the actual models are saved in the local server filesystems,
    where the filename of the models and the path are defined
    by db_model_path and fname here.

Parsing the message

-   The \_parse_message function just extracts the parameters from the
    received message:

> def \_parse_message(self, msg: List\[Any\]):
>
>     component_id = msg\[int(DBPushMsgLocation.component_id)\]
>
>     r = msg\[int(DBPushMsgLocation.round)\]
>
>     mt = msg\[int(DBPushMsgLocation.model_type)\]
>
>     model_id = msg\[int(DBPushMsgLocation.model_id)\]
>
>     gene_time = msg\[int(DBPushMsgLocation.gene_time)\]
>
>     meta_data = msg\[int(DBPushMsgLocation.meta_data)\]
>
>     local_prfmc = 0.0
>
>     if mt == ModelType.local:
>
>         try: local_prfmc = meta_data\[\"accuracy\"\]
>
>         except: pass
>
>     num_samples = 0
>
>     try: num_samples = meta_data\[\"num_samples\"\]
>
>     except: pass
>
>     return component_id, r, mt, model_id, gene_time, \\
>
>                                    local_prfmc, num_samples

CopyExplain

-   This function parses the received message into parameters related to
    agent ID or aggregator ID (component_id), round number (r), message
    type (mt), model_id, time of generation of the models (gene_time),
    and performance data as a dictionary format (meta_data).

-   The local performance data, local_prfmc, is extracted when the model
    type is local. The amount of sample data used at the local device is
    also extracted from meta_dect.

-   All these extracted parameters are returned at the end.

-   In the following section, we will explain the database
    implementation using the SQLite framework.

Defining the database with SQLite

-   The sqlite_db.py code creates the SQLite database and deals with
    storing and retrieving data from the database.

-   Importing libraries for the SQLite database

-   sqlite_db.py imports the basic general libraries and ModelType as
    follows:

> import sqlite3
>
> import datetime
>
> import logging
>
> from fl_main.lib.util.states import ModelType

CopyExplain

-   The ModelType from lib/util defines the type of the models: local
    models and (global) cluster models.

-   Defining and initializing the SQLiteDBHandler class

-   Then, the following code related to
    the SQLiteDBHandler class creates and initializes the SQLite
    database and inserts models into the SQLite database:

> class SQLiteDBHandler:
>
>     \"\"\"
>
>     SQLiteDB Handler class that creates and initialize
>
>     SQLite DB, and inserts models to the SQLiteDB
>
>     \"\"\"

CopyExplain

-   The initialization is very simple -- just setting
    the db_file parameter passed from the PseudoDB instance
    to self.db_file:

> def \_\_init\_\_(self, db_file):
>
>     self.db_file = db_file

CopyExplain

Initializing the database

-   In the following initialize_DB function, the database tables are
    defined with local and cluster models using SQLite (sqlite3):

> def initialize_DB(self):
>
>     conn = sqlite3.connect(f\'{self.db_file}\')
>
>     c = conn.cursor()
>
>     c.execute(\'\'\'CREATE TABLE local_models(model_id, \\
>
>         generation_time, agent_id, round, performance, \\
>
>         num_samples)\'\'\')
>
>     c.execute(\'\'\'CREATE TABLE cluster_models(model_id, \\
>
>         generation_time, aggregator_id, round, \\
>
>         num_samples)\'\'\')
>
>     conn.commit()
>
>     conn.close()

CopyExplain

-   The tables are simplified in this example so that you can easily
    follow the uploaded local models and their performance as well as
    the global models created by an aggregator.

-   The local_models table has a model ID (model_id), the time the model
    is generated (generation_time), an agent ID uploaded of the local
    model (agent_id), round information (round), the performance data of
    the local model (performance), and the number of samples used for
    FedAvg aggregation (num_samples).

-   cluster_models has a model ID (model_id), the time the model is
    generated (generation_time), an aggregator ID (aggregator_id), round
    information (round), and the number of samples (num_samples).

Inserting an entry into the database

-   The following code for insert_an_entry inserts the data received as
    parameters using sqlite3 libraries:

> def insert_an_entry(self, component_id: str, r: int, mt: \\
>
>     ModelType, model_id: str, gtime: float, local_prfmc: \\
>
>     float, num_samples: int):
>
>     conn = sqlite3.connect(self.db_file)
>
>     c = conn.cursor()
>
>     t = datetime.datetime.fromtimestamp(gtime)
>
>     gene_time = t.strftime(\'%m/%d/%Y %H:%M:%S\')
>
>     if mt == ModelType.local:
>
>         c.execute(\'\'\'INSERT INTO local_models VALUES \\
>
>         (?, ?, ?, ?, ?, ?);\'\'\', (model_id, gene_time, \\
>
>         component_id, r, local_prfmc, num_samples))
>
>     elif mt == ModelType.cluster:
>
>         c.execute(\'\'\'INSERT INTO cluster_models VALUES \\
>
>         (?, ?, ?, ?, ?);\'\'\', (model_id, gene_time, \\
>
>         component_id, r, num_samples))
>
>     conn.commit()
>
>     conn.close()

CopyExplain

-   This function takes the parameters of component_id (agent ID or
    aggregator ID), round number (r), message type (mt), model ID
    (model_id), the time the model is generated (gtime), the local
    model's performance data (local_prfmc), and the number of samples
    (num_samples) to insert an entry with the execute function of the
    SQLite library.

-   If the model type is *local*, the information of the models is
    inserted into the local_models table. If the model type
    is *cluster*, the information of the models is inserted into
    the cluster_models table.

-   Other functions, such as updating and deleting data from the
    database, are not implemented in this example code and it's up to
    you to write those additional functions.

-   In the following section, we will explain how to run the database
    server.

Running the database server

-   Here is the code for running the database server with the SQLite
    database:

> if \_\_name\_\_ == \"\_\_main\_\_\":
>
>     pdb = PseudoDB()
>
>     init_db_server(pdb.handler, pdb.db_ip, pdb.db_socket)

CopyExplain

-   The instance of PseudoDB class is created as pdb. The pdb.handler,
    the database's IP address (pdb.db_ip), and the database socket
    (pdb.db_socket) are used to start the process of receiving local and
    cluster models from an aggregator enabled by init_db_server from
    the communication_handler library in the util/lib folder.

-   Now, we understand how to implement and run the database server. The
    database tables and schema discussed here are minimally designed so
    that we can understand the fundamentals of the FL server's
    procedure. In the following section, we will discuss potential
    enhancements to the FL server.

FL Server Potential enhancements

-   Here are some of the key potential enhancements to the FL server
    discussed Here.

Redesigning the database

-   The database was intentionally designed with minimal table
    information in this book and needs to be extended, such as by having
    tables of the aggregator itself, agents, the initial base model, and
    the project info, among other things, in the database.

-   For example, the FL system described here does not support the
    termination and restart of the server and agent processes.

-   Thus, the FL server implementation is not complete, as it loses most
    of the information when any of the systems is stopped or failed.

Automating the registry of an initial model

-   In order to simplify the explanation of the process of registering
    the initial model, we defined the layers of the ML models using
    model names.

-   This registration of the model in the system can be automated so
    that just loading a certain ML model, such as PyTorch or Keras
    models, with file extensions such as .pt/.pth and .h5, will be
    enough for the users of the FL systems to start the process.

Performance metrics for local and global models

-   Again, to simplify the explanation of the FL server and the
    database-side functionalities, an accuracy value is just used as one
    of the performance criteria of the models.

-   Usually, ML applications have many more metrics to keep track of as
    performance data and they need to be enhanced together with the
    database and communications protocol design.

Fine-tuned aggregation

-   In order to simplify the process of aggregating the local models, we
    just used FedAvg, a weighted averaging method.

-   The number of samples can dynamically change depending on the local
    environment, and that aspect is enhanced by you.

-   There are also a variety of model aggregation methods, which will be
    explained later, *Model Aggregation*, of this work so that you can
    accommodate the best aggregation method depending on the ML
    applications to be created and integrated into the FL system.

Summary

-   Here, the basics and principles of FL server-side implementation
    were explained with actual code examples. Having followed the
    contents of This section, you should now be able to construct the FL
    server-side functionalities with model aggregation mechanisms.

-   The server-side components that were introduced here involve basic
    communications and the registration of the agents and initial
    models, managing state information used for the aggregation, and the
    aggregation mechanisms for creating the global cluster models.

-   In addition, we discussed the implementation of the database to just
    store the information of the ML models.

-   The code was simplified so that you were able to understand the
    principles of server-side functionalities. Further enhancements to
    many other aspects of constructing a more sustainable, resilient,
    and scalable FL system are up to you.

-   In the next section, we will discuss the principle of implementing
    the functionalities of the FL client and agent. The client side
    needs to provide some well-designed APIs for the ML applications for
    plugin use. Therefore, the section will discuss the FL client\'s
    core functionalities and libraries as well as the library
    integration into the very simple ML applications to enable the whole
    FL process.