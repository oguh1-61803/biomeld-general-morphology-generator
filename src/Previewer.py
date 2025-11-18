# All necessary libraries.
from lxml import etree
import plotly.express
import pandas
import pickle
import numpy


# This class shows a preview of the morphology.
class Previewer:

    MORPHOLOGY_PATH = "morphologies/morphology_"
    MORPHOLOGY_FILENAME = "morphology.vxa"
    MORPHOLOGY_CPPN_FILENAME = "cppn.pickle"

    X_VOXELS = "X_Voxels"
    Y_VOXELS = "Y_Voxels"
    Z_VOXELS = "Z_Voxels"

    MAPPING_REFERENCE = 0.5
    NO_VOXEL_VALUE = "0"
    PASSIVE_VOXEL_CHARACTER = "1"
    CONTRACTILE_VOXEL_CHARACTER = "3"

    def __init__(self):

        pass

    # This method triggers all the processes involved to generate the morphology preview.
    def create_morphology_preview(self, morphology_id):

        x_vector, y_vector, z_vector = self.__x_y_z_vectors(morphology_id)
        x, y, z, morph_data = self.__get_data_for_preview(morphology_id, x_vector, y_vector, z_vector)

        environment = {"x":x, "y":y, "z":z, "Material id": morph_data}
        morphology_data = pandas.DataFrame(data=environment)
        fig = plotly.express.scatter_3d(morphology_data, x="x", y="y", z="z",
                                        color="Material id", color_discrete_map={"0": "rgba(254, 254, 254, 0.0)",
                                                                                 "1":"rgba(6, 242, 247, 1.0)",
                                                                                 "3":"rgba(247, 6, 6, 1.0)"},
                                        title="Morphology " + str(morphology_id))
        fig.show()

    # This auxiliar method generates the layout space where the morphology will be drawn.
    def __x_y_z_vectors(self, morphology_id):

        morphology_path = self.MORPHOLOGY_PATH + str(morphology_id) + "/" + self.MORPHOLOGY_FILENAME
        parser = etree.XMLParser(remove_blank_text=True)
        raw_tree = etree.parse(morphology_path, parser=parser)
        root = raw_tree.getroot()

        x_vector = [float(x) for x in range(0, int(root.find("VXC").find("Structure").find(self.X_VOXELS).text))]
        y_vector = [float(y) for y in range(0, int(root.find("VXC").find("Structure").find(self.Y_VOXELS).text))]
        z_vector = [float(z) for z in range(0, int(root.find("VXC").find("Structure").find(self.Z_VOXELS).text))]

        return x_vector, y_vector, z_vector

    # This auxiliar method queries the CPPN to generate the morphology.
    def __get_data_for_preview(self, morphology_id, x_vector, y_vector, z_vector):

        with open(self.MORPHOLOGY_PATH + str(morphology_id) + "/" + self.MORPHOLOGY_CPPN_FILENAME, "rb" ) as pickle_file:

            cppn = pickle.load(pickle_file)

        x_data = []
        y_data = []
        z_data = []
        morph_data = []

        for z in z_vector:

            for y in y_vector:

                for x in x_vector:

                    cpp_input = [x, y, z]
                    cppn_output = cppn.activate(cpp_input)

                    vp = numpy.fabs(cppn_output[0])

                    if vp < self.MAPPING_REFERENCE:

                        morph_data.append(self.NO_VOXEL_VALUE)

                    else:

                        m = numpy.fabs(cppn_output[1])

                        if m < self.MAPPING_REFERENCE:

                            morph_data.append(self.PASSIVE_VOXEL_CHARACTER)

                        else:

                            morph_data.append(self.CONTRACTILE_VOXEL_CHARACTER)

                    x_data.append(x)
                    y_data.append(y)
                    z_data.append(z)

        return x_data, y_data, z_data, morph_data,


if __name__ == '__main__':

    # Before running this code, generate the morphologies (i.e., run Main.py). The previewer receives one parameter,
    # which represents the id of the morphology. For instance, once the morphologies are generated, and the
    # visualisation of the morphology 4 is required, pass a 4 (integer type) to the method "create_morphology_preview".
    p = Previewer()
    p.create_morphology_preview(1)