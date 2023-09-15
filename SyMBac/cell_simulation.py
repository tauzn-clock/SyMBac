import os
os.environ["LANG"]="en_US"
import pyglet
pyglet.options["shadow_window"] = False
import pickle
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from SyMBac.cell import Cell, StaticCell
from SyMBac.trench_geometry import trench_creator, get_trench_segments
import pymunk

from tqdm.auto import tqdm
import time
def run_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv, gravity,
                   phys_iters, max_length_var, width_var, save_dir, lysis_p=0, show_window = True, streamlit_mode = False):
    """
    Runs the rigid body simulation of bacterial growth based on a variety of parameters. Opens up a Pyglet window to
    display the animation in real-time. If the simulation looks bad to your eye, restart the kernel and rerun the
    simulation. There is currently a bug where if you try to rerun the simulation in the same kernel, it will be
    extremely slow.

    Parameters
    ----------

    trench_length : float
        Length of a mother machine trench (micron)
    trench_width : float
        Width of a mother machine trench (micron)
    cell_max_length : float
        Maximum length a cell can reach before dividing (micron)
    cell_width : float
        the average cell width in the simulation (micron)
    pix_mic_conv : float
        The micron/pixel size of the image
    gravity : float
        Pressure forcing cells into the trench. Typically left at zero, but can be varied if cells start to fall into
        each other or if the simulation behaves strangely.
    phys_iters : int
        Number of physics iterations per simulation frame. Increase to resolve collisions if cells are falling into one
        another, but decrease if cells begin to repel one another too much (too high a value causes cells to bounce off
        each other very hard). 20 is a good starting point
    max_length_var : float
        Variance of the maximum cell length
    width_var : float
        Variance of the maximum cell width
    save_dir : str
        Location to save simulation output
    lysis_p : float
        probability of cell lysis

    Returns
    -------
    cell_timeseries : lists
        A list of parameters for each cell, such as length, width, position, angle, etc. All used in the drawing of the
        scene later
    space : a pymunk space object
        Contains the rigid body physics objects which are the cells.
    """

    space = create_space()
    space.gravity = 0, gravity  # arbitrary units, negative is toward trench pole
    #space.iterations = 1000
    #space.damping = 0
    #space.collision_bias = 0.0017970074436457143*10
    space.collision_slop = 0.
    dt = 1 / 20  # time-step per frame
    pix_mic_conv = 1 / pix_mic_conv  # micron per pixel
    scale_factor = pix_mic_conv * 3  # resolution scaling factor

    trench_length = trench_length * scale_factor
    trench_width = trench_width * scale_factor
    trench_creator(trench_width, trench_length, (35, 0), space)  # Coordinates of bottom left corner of the trench

    cell1 = Cell(
        length=cell_max_length * scale_factor,
        width=cell_width * scale_factor,
        resolution=60,
        position=(20 + 35, 10),
        angle=0.8,
        space=space,
        dt= dt,
        growth_rate_constant=1,
        max_length=cell_max_length * scale_factor,
        max_length_mean=cell_max_length * scale_factor,
        max_length_var=max_length_var * np.sqrt(scale_factor),
        width_var=width_var * np.sqrt(scale_factor),
        width_mean=cell_width * scale_factor,
        parent=None,
        lysis_p=lysis_p
    )

    if show_window:
        from pymunk.pyglet_util import DrawOptions
        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)

        # key press event
        @window.event
        def on_key_press(symbol, modifier):

            # key "E" get press
            if symbol == pyglet.window.key.E:
                # close the window
                window.close()

    #global cell_timeseries
    #global x

    #try:
    #    del cell_timeseries
    #except:
    #    pass
    #try:
    #    del x
    #except:
    #    pass

    x = [0]
    cell_timeseries = []
    cells = [cell1]
    if show_window:
        pyglet.clock.schedule_interval(step_and_update, interval=dt, cells=cells, space=space, phys_iters=phys_iters,
                                       ylim=trench_length, cell_timeseries=cell_timeseries, x=x, sim_length=sim_length,
                                       save_dir=save_dir)
        pyglet.app.run()
    else:
        if streamlit_mode:
            import streamlit as st
            progress_text = "Simulation running"
            my_bar = st.progress(0, text=progress_text)
        for _ in tqdm(range(sim_length+2)):
            step_and_update(
                dt=dt, cells=cells, space=space, phys_iters=phys_iters, ylim=trench_length,
                cell_timeseries=cell_timeseries, x=x, sim_length=sim_length, save_dir=save_dir
            )
            if streamlit_mode:
                my_bar.progress((_)/sim_length, text=progress_text)

    # window.close()
    # phys_iters = phys_iters
    # for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
    #    cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length*1.1, cell_timeseries = cell_timeseries, x=x, sim_length = sim_length, save_dir = save_dir)
    #    if x > 250:
    #        cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def run_kinematics_simulation(trench_length, trench_width, cell_max_length, cell_width, sim_length, pix_mic_conv,
                    max_length_var, width_var, lysis_p, save_dir, show_window):
    
    
    dt = 0.1  # time-step per frame
    pix_mic_conv = 1 / pix_mic_conv  # micron per pixel
    scale_factor = pix_mic_conv * 3  # resolution scaling factor

    x = [0]
    cell_timeseries = []

    cell_timeseries, space = bulk_fall_in(
                show_window, dt, cell_max_length, scale_factor, trench_length, trench_width, cell_width, max_length_var, width_var, lysis_p=lysis_p, ylim=trench_length*scale_factor,
                cell_timeseries=cell_timeseries, x=x, sim_length=sim_length, save_dir=save_dir
            )
    '''
    if show_window:

        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)

        # key press event
        @window.event
        def on_key_press(symbol, modifier):

            # key "E" get press
            if symbol == pyglet.window.key.E:
                # close the window
                window.close()
        pyglet.clock.schedule_interval(bulk_fall_in, interval=dt, cells=cells, space=space, cell_max_length=cell_max_length, scale_factor=scale_factor, trench_length=trench_length, trench_width = trench_width, cell_width=cell_width, max_length_var=max_length_var, width_var=width_var, lysis_p=lysis_p,
                                       ylim=trench_length, cell_timeseries=cell_timeseries, x=x, sim_length=sim_length,
                                       save_dir=save_dir)
        pyglet.app.run()
    else:
        for _ in tqdm(range(sim_length+2)):
            cells = []
    '''
            
    
    # window.close()
    # phys_iters = phys_iters
    # for x in tqdm(range(sim_length+250),desc="Simulation Progress"):
    #    cells = step_and_update(dt=dt, cells=cells, space=space, phys_iters=phys_iters,ylim=trench_length*1.1, cell_timeseries = cell_timeseries, x=x, sim_length = sim_length, save_dir = save_dir)
    #    if x > 250:
    #        cell_timeseries.append(deepcopy(cells))
    return cell_timeseries, space

def insert_cell(dt, cells, cell_max_length, scale_factor, trench_length, trench_width, cell_width, max_length_var, width_var, lysis_p, space, ylim, cell_timeseries):
        max_cell_number = int(trench_length*trench_width/(cell_max_length*cell_width))
        #change the distribution to distribution obtained from charlie's data.
        _length = np.random.normal(cell_max_length * scale_factor, np.sqrt(max_length_var)*scale_factor)
        _width = np.random.normal(cell_width * scale_factor, np.sqrt(width_var)*scale_factor)
        _diagonal_length = np.sqrt(_length**2 + _width**2)
        _max_allowed_angle = np.arcsin(10/_diagonal_length)
        
        for shape in space.shapes:
            if shape.body.position.y < 0 or shape.body.position.y > ylim:
                space.remove(shape.body, shape)
                space.step(dt)
        #import timesadgf((2*abs(_max_allowed_angle)) * np.random.random_sample() - _max_allowed_angle)
        cell = StaticCell(
            length=_length,
            width=_width,
            resolution=60,
            lysis_p = lysis_p,
            position=(20 + np.random.normal(35, 5), ylim - 1),
            angle=(2*abs(_max_allowed_angle)) * np.random.random_sample() - _max_allowed_angle + np.pi/4,
            space=space,
            dt= dt,
        )
        print(len(cells))
        cells.append(cell)
        #print(len(cells))
        wipe_space(space)
        if cell.shape.body.position.y < 0 or cell.shape.body.position.y > ylim or cell.shape.body.position.x < 20 or cell.shape.body.position.x > 20 + 70:            #graveyard.append([cell, "outside"])
            cells.remove(cell)
            space.step(dt)

        elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(cells) > 1:   # in case all cells disappear
            #graveyard.append([cell, "lysis"])
            cells.remove(cell)
            space.step(dt)

        update_pm_cells(cells)

        prev_cell_poses = np.array([[cell.body.position.x, cell.body.position.y] for cell in cells])
        for _ in range(100):
            #update_pm_cells(cells)
            space.step(dt)

        update_cell_positions(cells)
        current_cell_poses = np.array([[cell.body.position.x, cell.body.position.y] for cell in cells])
        #print(np.linalg.norm(current_cell_poses - prev_cell_poses)/np.linalg.norm(prev_cell_poses))
        while np.linalg.norm(current_cell_poses - prev_cell_poses)/(np.linalg.norm(prev_cell_poses)+1e-5) > 0.1:
            #update_pm_cells(cells)
            #print([cell.body.velocity.length for cell in cells])
            #print([cell.body.position for cell in cells])
            update_pm_cells(cells)
            update_cell_positions(cells)
            prev_cell_poses = current_cell_poses
            space.step(dt)
            current_cell_poses = np.array([[cell.body.position.x, cell.body.position.y] for cell in cells])
            for i,_cell in enumerate(cells):
                if _cell.shape.body.position.y < 0 or _cell.shape.body.position.y > ylim or _cell.shape.body.position.x < 20 or _cell.shape.body.position.x > 20 + 70:            #graveyard.append([cell, "outside"])
                    cells.remove(_cell)
                    current_cell_poses = np.delete(current_cell_poses, i, 0)
                    prev_cell_poses = np.delete(prev_cell_poses, i, 0)
                    if len(cells) >= 2:
                        if _cell.shape.body.position.y > ylim and cells[-1].shape.body.position.y > ylim - _length:
                            cells.append(_cell)
                            print(len(cells))
                            print("cell overflow")
                            cell_timeseries.append(deepcopy(cells))
                            pyglet.app.exit()
                            cells = []
                            #window.close()
                            return True, cells
                    else:
                        pass
            if len(cells) == max_cell_number:
                print("more cells than possible")
                print(len(cells))
                cell_timeseries.append(deepcopy(cells))
                pyglet.app.exit()
                cells = []
                #window.close()
                return True, cells   
            if len(cells) >= 50:
                break
                         
        update_cell_positions(cells)
        return None, cells


def bulk_fall_in(show_window, dt, cell_max_length, scale_factor, trench_length, trench_width, cell_width, max_length_var, width_var, lysis_p, ylim, cell_timeseries, x, sim_length, save_dir):
    if show_window:
        from pymunk.pyglet_util import DrawOptions
        window = pyglet.window.Window(700, 700, "SyMBac", resizable=True)
        options = DrawOptions()
        options.shape_outline_color = (10,20,30,40)
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options)

        # key press event
        @window.event
        def on_key_press(symbol, modifier):

            # key "E" get press
            if symbol == pyglet.window.key.E:
                # close the window
                window.close()
    #new_cells = []
    #graveyard = []
    space = create_space()
    space.gravity = 0, -10  # arbitrary units, negative is toward trench pole
    #space.iterations = 1000
    #space.damping = 0
    #space.collision_bias = 0.0017970074436457143*10
    space.collision_slop = 0.
    trench_creator(trench_width*scale_factor, trench_length*scale_factor, (35, 0), space)  # Coordinates of bottom left corner of the trench

    cells = []

    while x[0] < sim_length - 2:
        

        print("iteration: " + str(x[0]))
        if show_window:

            pyglet.clock.schedule_interval(insert_cell, interval = dt, cells = cells, cell_max_length = cell_max_length, scale_factor = scale_factor, trench_length = trench_length, trench_width = trench_width, cell_width = cell_width, max_length_var = max_length_var, width_var = width_var, lysis_p = lysis_p, space = space, ylim=  ylim, cell_timeseries = cell_timeseries)
            pyglet.app.run() 
            #window.close()
            #time.sleep(1)
            #window.close()
            #wipe_space(space)
            #del window
            #del options
            #del space
            #del cells
        else:
            completed = False
            while not completed:
                completed, cells = insert_cell(dt, cells, cell_max_length, scale_factor, trench_length, trench_width, cell_width, max_length_var, width_var, lysis_p, space, ylim, cell_timeseries)

        x[0] += 1
    pyglet.app.exit()
    if show_window:
        window.close()
    with open(save_dir+"/cell_timeseries.p", "wb") as f:
        pickle.dump(cell_timeseries, f)
    with open(save_dir+"/space_timeseries.p", "wb") as f:
        pickle.dump(space, f)

    return cell_timeseries, space


def create_space():
    """
    Creates a pymunk space

    :return pymunk.Space space: A pymunk space
    """

    space = pymunk.Space(threaded=False)
    #space.threads = 2
    return space

def update_cell_lengths(cells):
    """
    Iterates through all cells in the simulation and updates their length according to their growth law.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
    """
    for cell in cells:
        cell.update_length()


def update_pm_cells(cells):
    """
    Iterates through all cells in the simulation and updates their pymunk body and shape objects. Contains logic to
    check for cell division, and create daughters if necessary.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.

    """
    for cell in cells:
        if cell.is_dividing():
            daughter_details = cell.create_pm_cell()
            if len(daughter_details) > 2: # Really hacky. Needs fixing because sometimes this returns cell_body, cell shape. So this is a check to ensure that it's returing daughter_x, y and angle
                daughter = Cell(**daughter_details)
                cell.daughter = daughter
                cells.append(daughter)
        else:
            cell.create_pm_cell()

def update_cell_positions(cells):
    """
    Iterates through all cells in the simulation and updates their positions, keeping the cell object's position
    synchronised with its corresponding pymunk shape and body inside the pymunk space.

    :param list(SyMBac.cell.Cell) cells: A list of all cells in the current timepoint of the simulation.
    """
    for cell in cells:
        cell.update_position()

def wipe_space(space):
    """
    Deletes all cells in the simulation pymunk space.

    :param pymunk.Space space:
    """
    for body, poly in zip(space.bodies, space.shapes):
        if body.body_type == 0:
            space.remove(body)
            space.remove(poly)

def update_cell_parents(cells, new_cells):
    """
    Takes two lists of cells, one in the previous frame, and one in the frame after division, and updates the parents of
    each cell

    :param list(SyMBac.cell.Cell) cells:
    :param list(SyMBac.cell.Cell) new_cells:
    """
    for i in range(len(cells)):
        cells[i].update_parent(id(new_cells[i]))

def step_and_update(dt, cells, space, phys_iters, ylim, cell_timeseries,x,sim_length,save_dir):
    """
    Evolves the simulation forward

    :param float dt: The simulation timestep
    :param list(SyMBac.cell.Cell)  cells: A list of all cells in the current timestep
    :param pymunk.Space space: The simulations's pymunk space.
    :param int phys_iters: The number of physics iteration in each timestep
    :param int ylim: The y coordinate threshold beyond which to delete cells
    :param list cell_timeseries: A list to store the cell's properties each time the simulation steps forward
    :param int list: A list with a single value to store the simulation's progress.
    :param int sim_length: The number of timesteps to run.
    :param str save_dir: The directory to save the simulation information.

    Returns
    -------
    cells : list(SyMBac.cell.Cell)

    """
    for shape in space.shapes:
        if shape.body.position.y < 0 or shape.body.position.y > ylim:
            space.remove(shape.body, shape)
            space.step(dt)
    #new_cells = []
    #graveyard = []
    for cell in cells:
        if cell.shape.body.position.y < 0 or cell.shape.body.position.y > ylim:
            #graveyard.append([cell, "outside"])
            cells.remove(cell)
            space.step(dt)
        elif norm.rvs() <= norm.ppf(cell.lysis_p) and len(cells) > 1:   # in case all cells disappear
            #graveyard.append([cell, "lysis"])
            cells.remove(cell)
            space.step(dt)
        else:
            pass
            #new_cells.append(cell)
    #cells = deepcopy(new_cells)
    #graveyard = deepcopy(graveyard)

    wipe_space(space)

    update_cell_lengths(cells)
    update_pm_cells(cells)

    for _ in range(phys_iters):
        space.step(dt)
    update_cell_positions(cells)

    #print(str(len(cells))+" cells")
    if x[0] > 1:
        #copy_cells = deepcopy(cells)

        cell_timeseries.append(deepcopy(cells))
        copy_cells = cell_timeseries[-1]
        update_cell_parents(cells, copy_cells)
        #del copy_cells
    if x[0] == sim_length-1:
        with open(save_dir+"/cell_timeseries.p", "wb") as f:
            pickle.dump(cell_timeseries, f)
        with open(save_dir+"/space_timeseries.p", "wb") as f:
            pickle.dump(space, f)
        pyglet.app.exit()
        return cells
    x[0] += 1
    return (cells)

