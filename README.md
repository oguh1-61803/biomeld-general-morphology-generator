> # General morphology generator

Morphology generator based on NEAT. This code generates morphologies employing CPPNs, which are generated using the neat-python library.

> **Requirements**

The code was written in Python, version 3.11. The libraries utilised are as follows:

* ConfigUpdater -- 3.2
* lxml -- 6.0.2
* neat-python-2023 -- 0.93
* numpy -- 2.3.4
* scipy -- 1.16.3

> **CPPNs**

In this approach, CPPNs have three inputs (x,y,z), representing the coordinates of the i point in the tridimensional layout where morphologies are designed. Furthermore, it has two outputs. The first defines the presence or not of a voxel in the i point of the layout, and the second output defines the material type the voxel represents. There are two types of materials: passive and contractile, encoded as 1 and 3, respectively. In general: 

CPPN(x,y,z) = (presence, material type).
  
> **Output**

The code generates three files per CPPN:

1. A .pickle file containing the serialised CPPN object.
   
2. A .vxa file, containing the morphology designed by the CPPN. The file has the format required by VoxCAD the visual version of Voxelyze for morphology visualisation.
   
3. A .txt file, containing the topology of the CPPN. The nomanclature utilised in the file is as follows:
   * Nodes refers to neurons.
   * Negative ids represent input neurons. In this approach -1, -2, -3 are the ids of the three input neurons.
   * Ids 0 and 1 are for the output neurons.
   * The rest of ids are associated to hidden neurons.
   * Connections show the id of the "origin" node and the id of "destination" node.

The output of the code is saved in a folder called "morphologies".

> **Assumptions**
  
* The specific parameters related to NEAT are taken from the configuration of previous reseearch and have not been modified.

> **Other considerations**

* The code has been designed to be wrapped in a REST API if needed. The main input is a JSON contaning relevant data. See the Main.py file for more detail.
  
* In order to visualise the morphologies, VoxCAD is needed: https://github.com/skriegman/reconfigurable_organisms .
  
* More technical details (e.g., how to add custom activation functions to de dictionary) can be found as comments within the code.

