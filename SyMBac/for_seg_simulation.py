import numpy as np
from joblib import Parallel, delayed

from SyMBac.cell_simulation import run_kinematics_simulation
from SyMBac.drawing import draw_scene, get_space_size, gen_cell_props_for_draw, generate_curve_props
from SyMBac.trench_geometry import  get_trench_segments
from tqdm.autonotebook import tqdm
import napari
import pickle

class ForSegSimulation:

    def __init__(self, trench_length, trench_width, cell_max_length, max_length_var, cell_width, width_var, lysis_p, opacity_var, sim_length, pix_mic_conv, resize_amount, save_dir, show_window):
        self.trench_length = trench_length
        self.trench_width = trench_width
        self.cell_max_length = cell_max_length
        self.max_length_var = max_length_var
        self.cell_width = cell_width
        self.width_var = width_var
        self.lysis_p = lysis_p
        self.opacity_var = opacity_var
        self.sim_length = sim_length
        self.pix_mic_conv = pix_mic_conv
        self.resize_amount = resize_amount
        self.save_dir = save_dir
        self.offset = 50
        self.show_window = show_window

    def run_simulation(self):
        self.cell_timeseries, self.space = run_kinematics_simulation(
            trench_length=self.trench_length,
            trench_width=self.trench_width,
            cell_max_length=self.cell_max_length,  # 6, long cells # 1.65 short cells
            cell_width=self.cell_width,  # 1 long cells # 0.95 short cells
            sim_length=self.sim_length,
            pix_mic_conv=self.pix_mic_conv,
            max_length_var=self.max_length_var,
            width_var=self.width_var,
            lysis_p= self.lysis_p,
            save_dir=self.save_dir,
            show_window=self.show_window,
        )

    def draw_simulation_OPL(self, do_transformation = True, label_masks = True, dynamics_free = False, return_output = False, streamlit_mode=False, timeseries_repo = None):
        if streamlit_mode:
            from stqdm import stqdm as tqdm
        else:
            from tqdm.autonotebook import tqdm

        """
        Draw the optical path length images from the simulation. This involves drawing the 3D cells into a 2D numpy
        array, and then the corresponding masks for each cell.

        After running this function, the Simulation object will gain two new attributes: ``self.OPL_scenes`` and ``self.masks`` which can be accessed separately.

        :param bool do_transformation: Sets whether to transform the cells by bending them. Bending the cells can add realism to a simulation, but risks clipping the cells into the mother machine trench walls.

        :param bool label_masks: Sets whether the masks should be binary, or labelled. Masks should be binary is training a standard U-net, such as with DeLTA, but if training Omnipose (recommended), then mask labelling should be set to True.

        :param bool return_output: Controls whether the function returns the OPL scenes and masks. Does not affect the assignment of these attributes to the instance.

        Returns
        -------
        output : tuple(list(numpy.ndarray), list(numpy.ndarray))
           If ``return_output = True``, a tuple containing lists, each of which contains the entire simulation. The first element in the tuple contains the OPL images, the second element contains the masks

        """
        if timeseries_repo is not None:
            
            with (open(timeseries_repo + "/space_timeseries.p", "rb")) as openfile:
                while True:
                    try:
                        _space = pickle.load(openfile)
                        print(pickle.load(openfile))
                    except EOFError:
                        break

            with (open(timeseries_repo + "/cell_timeseries.p", "rb")) as openfile:
                while True:
                    try:
                        _cell_timeseries = pickle.load(openfile)
                    except EOFError:
                        break
        else:
            _space = self.space
            _cell_timeseries = self.cell_timeseries
        self.main_segments = get_trench_segments(_space)
        ID_props = generate_curve_props(_cell_timeseries)

        self.cell_timeseries_properties = Parallel(n_jobs=-1)(
            delayed(gen_cell_props_for_draw)(a, ID_props) for a in tqdm(_cell_timeseries, desc='Timeseries Properties'))

        space_size = get_space_size(self.cell_timeseries_properties)

        scenes = Parallel(n_jobs=-1)(delayed(draw_scene)(
        cell_properties, do_transformation, space_size, self.offset, label_masks, dynamics_free) for cell_properties in tqdm(
            self.cell_timeseries_properties, desc='Scene Draw:'))
        #print(scenes)
        self.OPL_scenes = [_[0] for _ in scenes]
        self.masks = [_[1] for _ in scenes]

        with open(self.save_dir+"/simulation.p", "wb") as f:
            pickle.dump(self, f)

        if return_output:
            return self.OPL_scenes, self.masks

    def visualise_in_napari(self):
        """
        Opens a napari window allowing you to visualise the simulation, with both masks, OPL images, interactively.
        :return:
        """
        viewer = napari.view_image(np.array(self.OPL_scenes), name='OPL scenes')
        viewer.add_labels(np.array(self.masks), name='Synthetic masks')
        napari.run()