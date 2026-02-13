# All the required imports and libraries.
from concurrent.futures import ProcessPoolExecutor
from src.coordinate_systems import CoordinateTransformer
from src.geometric_fitness import calculate_fitness, calculate_all_metrics
from lxml import etree
import pickle
import numpy
import neat
import os
import json




# This class manipulates CPPNs and generates the files associated to them.
class DesignEngine:

    MAPPING_REFERENCE = 0.5
    NO_VOXEL_CHARACTER = "0"
    PASSIVE_VOXEL_CHARACTER = "1"
    CONTRACTILE_VOXEL_CHARACTER = "3"

    X_VOXELS = "X_Voxels"
    Y_VOXELS = "Y_Voxels"
    Z_VOXELS = "Z_Voxels"

    BASE_FILE_PATH = "src/morphology_base.vxa"
    MORPHOLOGIES_PATH = "src/morphologies/morphology_"
    NUMBER_OF_WORKERS = 5

    def __init__(self, parameters_data):

        self.dimensions_3d_layout = parameters_data.get("dimensions_3d_layout")
        self.fitness_shape = parameters_data.get("fitness_shape")
        
        # NEW: Initialize coordinate transformer
        coordinate_system = parameters_data.get("coordinate_system", "cartesian")
        self.coordinate_transformer = CoordinateTransformer(
            coordinate_system,
            self.dimensions_3d_layout
        )
        
        print(f"[DesignEngine] Using coordinate system: {coordinate_system}")
        print(f"[DesignEngine] Dimensions: {self.dimensions_3d_layout}")

        self.x_vector = None
        self.y_vector = None
        self.z_vector = None

        self.__initialise_layout(parameters_data)
        self.generation_number = 0

    # This method builds CPPNs using the neat-python library.
    def build_morphologies_using_cppns(self, genomes, configuration):

        list_of_cppns = []
        morphology_counter = 0
        list_of_morphologies = []
        list_of_metrics = []

        for genome_id, genome in genomes:

            os.makedirs(self.MORPHOLOGIES_PATH + str(morphology_counter), exist_ok=True)

            # For each CPPN, a file containing its topology is saved.
            with open(self.MORPHOLOGIES_PATH + str(morphology_counter) + "/neurons_and_connections.txt", "w") as file:

                print("{!s}".format(genome), file=file)
                print("%s nodes with %s connections." % (genome.size()[0], genome.size()[1]), file=file)
                file.close()

            cppn = neat.nn.FeedForwardNetwork.create(genome, configuration)

            # Each CPPN, is saved in a file.
            
            with open(self.MORPHOLOGIES_PATH + str(morphology_counter) + "/cppn.pickle", "wb") as file:

                pickle.dump(cppn, file)
                file.close()

            list_of_cppns.append(cppn)
            #print("CCC2", self.generation_number, morphology_counter, genome_id)
            #print("CCCCCCC3", calculate_fitness(self.Morph2NP(self.build_morphology(cppn)),'cylinder'))
            morphology = self.build_morphology(cppn)
            genome.fitness = calculate_fitness(self.Morph2NP(morphology),'cylinder')
            morphology_counter += 1
            list_of_morphologies.append(morphology)
            list_of_metrics.append(calculate_all_metrics(self.Morph2NP(morphology)))

        #list_of_morphologies = []

        # The morphologies are built in parallel-like fashion.
        """
        with ProcessPoolExecutor(max_workers=self.NUMBER_OF_WORKERS) as executor:

            for morphology in executor.map(self.build_morphology, list_of_cppns, chunksize=2):

                list_of_morphologies.append(morphology)
        """

        cppn_ids = [x for x in range(len(list_of_morphologies))]
        #print("CCCCCCCCCC2", cppn_ids)

        # pop.statistics.best_genome()
        # The morphologies are saved (in parallel-like fashion) in a .vxa file, the format required by VoxCad,
        # the interactive version of Voxelyze.
        self.generation_number = self.generation_number + 1
        print("GGGGGNNNNN", self.generation_number)
        with ProcessPoolExecutor(max_workers=self.NUMBER_OF_WORKERS) as executor:

            executor.map(self.write_vxa_file, list_of_morphologies, cppn_ids, list_of_metrics, chunksize=2)

    # This method queries the CPPN, which is received as parameter, to design the morphology. CPPNs have three neurons
    # as inputs, representing the (x,y,z) coordinates. Furthermore, they have two neurons as outputs. The first one
    # defines the presence (or not) of a voxel. The second one, decides the type of material.
    def build_morphology(self, cppn):

        morphology = []

        for z_coordinate in self.z_vector:

            layer = ""

            for y_coordinate in self.y_vector:

                for x_coordinate in self.x_vector:

                    # NEW: Transform and normalize coordinates for CPPN
                    cppn_input = self.coordinate_transformer.transform_and_normalize(
                        x_coordinate, y_coordinate, z_coordinate
                    )
                    
                    cppn_output = cppn.activate(cppn_input)
                    voxel_presence = numpy.fabs(cppn_output[0])

                    if voxel_presence < self.MAPPING_REFERENCE:

                        layer += self.NO_VOXEL_CHARACTER

                    else:

                        material_type = numpy.fabs(cppn_output[1])

                        if material_type < self.MAPPING_REFERENCE:

                            layer += self.PASSIVE_VOXEL_CHARACTER

                        else:

                            layer += self.CONTRACTILE_VOXEL_CHARACTER

            morphology.append(layer)

        return morphology




    def NotBlankMorphology(self, _morphology) : 
        #print("PPPPPP2", _morphology)
        for z in _morphology:
            for xy in z:
                #print("PP3",xy)
                if  (xy != _morphology[0][0]) : return 1
        return 0





    def Morph2NPx(self,_morphology) : 
        #print("NNNNNNNNNNNNNNNN")
        cylinder_morph = numpy.zeros((15, 8, 10), dtype=int)
        #print("NNNNNNNNNNNNNNNN2")
        for z in range(0, 9):
            layerz = _morphology[z]
            for y in range(0, 7):
                for x in range(0, 14):
                    #print("NNN3",x,y,z)
                    cylinder_morph[x, y, z] = int(layerz[y*15 + x])
        return cylinder_morph


    def Morph2NP(self,_morphology) : 
        z_size = len(_morphology)
        #print("NNNNNNNNNNNNNNNN")
        cylinder_morph = numpy.zeros((z_size, 8, 15), dtype=int)
        #print("NNNNNNNNNNNNNNNN2")
        for z in range(0, z_size - 1):
            layerz = _morphology[z]
            for y in range(0, 7):
                for x in range(0, 14):
                    #print("NNN3",x,y,z)
                    cylinder_morph[z, y, x] = int(layerz[y*15 + x])
        return cylinder_morph


    """
    for z in range(60):
        for y in range(2, 6):
            for x in range(5, 10):
                # Hollow cylinder
                if not (3 <= y <= 4 and 6 <= x <= 8):
                    cylinder_morph[z, y, x] = 1
    """




    # This method writes the morphology generated by a CPPN in a .vxa file.
    def write_vxa_file(self, morphology, cppn_id, metric):

        # These two lines are helpful to trace the morphologies and identifying in which folder are saved.
        # Uncomment them if need it.
        print(cppn_id)
        print(morphology)




        if (self.NotBlankMorphology(morphology)) : 
            with open('log.html', 'a') as ff:
                morphNP = self.Morph2NP(morphology)
                print("CCCCCCCC")
                rs = calculate_fitness(morphNP,self.fitness_shape)
                print ("BBBBBBBBBB", rs)
                ff.write('"Gen num'+str(self.generation_number)+'_id_'+str(cppn_id)+'_fit_'+str(rs)+'":')
                #ff.write('"'+str(cppn_id)+'":')
                json.dump(morphology, ff)
                #self._morphology = morphology
                ff.write(',\n\r')


            with open('metrics.html', 'a') as ff:
                ff.write('"Gen num'+str(self.generation_number)+'_id_'+str(cppn_id)+'":{')
                #ff.write('"'+str(cppn_id)+'":')
                json.dump(metric, ff)
                #self._morphology = morphology
                ff.write('},\n\r')






        parser = etree.XMLParser(remove_blank_text=True)
        raw_tree = etree.parse(self.BASE_FILE_PATH, parser=parser)
        root = raw_tree.getroot()
        root.find("VXC").find("Structure").find(self.X_VOXELS).text = str(self.dimensions_3d_layout.get("x"))
        root.find("VXC").find("Structure").find(self.Y_VOXELS).text = str(self.dimensions_3d_layout.get("y"))
        root.find("VXC").find("Structure").find(self.Z_VOXELS).text = str(self.dimensions_3d_layout.get("z"))
        root.find("Simulator").find("GA").find("FitnessFileName").text = "fitness.xml"

        for child in root.find("VXC").find("Structure").find("Data"):

            root.find("VXC").find("Structure").find("Data").remove(child)

        for layer in morphology:

            l = etree.SubElement(root.find("VXC").find("Structure").find("Data"), "Layer")
            l.text = etree.CDATA(layer)

        with open(self.MORPHOLOGIES_PATH + str(cppn_id) + "/morphology.vxa", 'wb') as f:

             raw_tree.write(f, encoding="ISO-8859-1", pretty_print=True)

    # This auxiliar method initialises the vectors that are useful for building morphologies.
    def __initialise_layout(self, parameters_data):

        self.x_vector = [float(x) for x in range(0, parameters_data.get("dimensions_3d_layout").get("x"))]
        self.y_vector = [float(y) for y in range(0, parameters_data.get("dimensions_3d_layout").get("y"))]
        self.z_vector = [float(z) for z in range(0, parameters_data.get("dimensions_3d_layout").get("z"))]
