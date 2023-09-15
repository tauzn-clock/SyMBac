import sys
sys.path.insert(1, '/home/georgeos/Documents/GitHub/SyMBac/') # Not needed if you installed SyMBac using pip

from SyMBac.for_seg_simulation import ForSegSimulation
from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.misc import get_sample_images, misc_load_img

def main(initialized):
    real_image = misc_load_img("./00000.tif")

    my_simulation = ForSegSimulation(
        trench_length=15,
        trench_width=1.3,
        cell_max_length=4, #6, long cells # 1.65 short cells
        cell_width= 0.9, #1 long cells # 0.95 short cells
        opacity_var=0.1,
        sim_length = 100,
        pix_mic_conv = 0.065,
        max_length_var = 0.5,
        width_var = 0.01,
        lysis_p= 0.,
        save_dir="./",
        resize_amount = 3,
        show_window=False
    )

    my_simulation.run_simulation()

    my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=False, dynamics_free=True, timeseries_repo = "./")
    
    from PIL import Image
    import numpy as np
    for i in range(len(my_simulation.OPL_scenes)):
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2,5))
        #ax1.imshow(my_simulation.OPL_scenes[i], cmap="Greys_r")
        #ax1.axis("off")
        #ax2.imshow(my_simulation.masks[i])
        #ax2.axis("off")
        im = Image.fromarray(255*(my_simulation.OPL_scenes[i] - np.min(my_simulation.OPL_scenes[i]))/(np.max(my_simulation.OPL_scenes[i]) - np.min(my_simulation.OPL_scenes[i])))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("oplImage" + str(i) + ".png")
        im = Image.fromarray(255*(my_simulation.masks[i] - np.min(my_simulation.masks[i]))/(np.max(my_simulation.masks[i]) - np.min(my_simulation.masks[i])))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("mask" + str(i) + ".png")

    my_kernel = PSF_generator(
        radius = 50,
        wavelength = 0.75,
        NA = 1.2,
        n = 1.3,
        resize_amount = 3,
        pix_mic_conv = 0.065,
        apo_sigma = 20,
        mode="phase contrast",
        condenser = "Ph3")
    my_kernel.calculate_PSF()
    my_camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)
    #my_camera.render_dark_image(size=(300,300))
    
    import pickle
    with (open("." + "/simulation.p", "rb")) as openfile:
        try:
            _my_simulation = pickle.load(openfile)
        except EOFError:
            raise EOFError
    my_renderer = Renderer(simulation = _my_simulation, PSF = my_kernel, real_image = real_image, camera = my_camera)
    my_renderer.select_intensity_napari()
    if (initialized):
        my_renderer.update_synth_image_params(save_dir=".")
    else:    
        my_renderer.optimise_synth_image(manual_update=False, save_dir=".")
    my_renderer.generate_training_data(sample_amount=0.1, randomise_hist_match=True, randomise_noise_match=True, burn_in=40, n_samples = 500, save_dir="/tmp/test/", params_dir=".", in_series=False)

if __name__ == "__main__":
    main(initialized=False)