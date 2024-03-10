"""FFT mode solver for Fabry-Perot resonators

This module includes classes that can import and process measured mirror
profiles, as well as simulate modes and corresponding losses of the resonators 
formed by the mirrors.

Typical usage example:

  mirror1 = FPMirror('mirror1.datx')
  mirror1.level_and_center()
  
  resonator1 = FPResonator(
  mirror_a = mirror1,
  wavelength = 1550e-9,
  length = 500e-6,
  )
  resonator1.waist()
  resonator1.mode(sort = "loss")
  resonator1.plot_mode(mode_no = 0)

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from matplotlib import cm
from matplotlib.patches import Circle

from typing import Tuple

from datx2py import datx2py
from scipy.special import erf
import solver_core

# unit conversion table to meters
UNIT_TABLE = {'NanoMeters': 1.0e-9,
              'nm': 1.0e-9,
              'um': 1.0e-6,
              'mm': 1.0e-3,
              'm': 1.0}

# speed of light in meter per second
SPEED_OF_LIGHT = 3.0e8

class FPMirror:
    """Mirror profile import, storage and processing
    
    Attributes:
        z_raw: raw profile height data, values depend on base units
        x_raw, y_raw: coordinates of the raw data, base units are pixels
        x_vals, y_vals, z_vals: original data in the user-defined units
        x_vals_processed, y_vals_processed, z_vals_processed: 
        processed data in the user-defined units

    Todos:
        add a tilt method that can tilt the data
        add a regulate(?) method that can regulate the dataset size 
        for two mirror simulations
    """
    
    def __init__(
        self, 
        file_path: str = None, 
        data_x: np.ndarray = None,
        data_y: np.ndarray = None,
        data_z: np.ndarray = None,
        xy_unit: str = 'um', 
        z_unit: str = 'um'
    ) -> None:
        """Initializes the instance with mirror profile
        
        Args:
          file_path: path to the .datx mirror profile file
          if file_path is not provided, data_x, data_y and data_z should be 
          present:
          data_x, data_y, data_z: xy coordinates and z height of the mirror,
          should be in the nxn meshgrid form
          xy_unit: user-defined units along x and y axes
          z_unit: user-defined unit along z axis
        """
        # if file path is not provide, FPMirror will be constructed from user
        # -provided data input (Todo: convert this into a copy? method)
        if file_path == None:
            self.x_vals = data_x
            self.y_vals = data_y
            self.z_vals = data_z
            
            # set units
            self.xy_unit_symbol = xy_unit
            self.z_unit_symbol = z_unit
            self.xy_unit = UNIT_TABLE[self.xy_unit_symbol]
            self.z_unit = UNIT_TABLE[self.z_unit_symbol]
        
        else:
            # import .datx file
            h5data = datx2py(file_path)

            # parse out height values
            data = h5data['Data']['Surface']
            data = list(data.values())[0]
            z_raw = data['vals']
            z_raw[z_raw == data['attrs']['No Data']] = np.nan

            # get z unit and convert it to meters
            # then save original z data
            self.z_raw_unit_symbol = data['attrs']['Z Converter']['BaseUnit']
            self.z_raw_unit = UNIT_TABLE[self.z_raw_unit_symbol]
            self.z_raw = z_raw*self.z_raw_unit

            # save x and y resolutions, get their units and convert to meters
            # then generate x and y grid
            self.x_res_unit_symbol = data['attrs']['X Converter']['BaseUnit']
            self.x_res = data['attrs']['X Converter']['Parameters'][1]
            self.y_res_unit_symbol = data['attrs']['Y Converter']['BaseUnit']
            self.y_res = data['attrs']['Y Converter']['Parameters'][1]
            if self.x_res == self.y_res:
                self.xy_res = self.x_res
            else:
                raise ValueError("x and y resolutions are not identical!")

            y_len, x_len = np.shape(self.z_raw)
            y_i, x_j = np.meshgrid(
                np.arange(y_len-1, -1, -1), np.arange(x_len), indexing = 'ij'
            )
            self.x_raw = x_j*self.x_res
            self.y_raw = y_i*self.y_res
            
            # convert data into user-defined units
            self.xy_unit_symbol = xy_unit
            self.z_unit_symbol = z_unit
            self.xy_unit = UNIT_TABLE[self.xy_unit_symbol]
            self.z_unit = UNIT_TABLE[self.z_unit_symbol]
            
            self.x_vals = self.x_raw/self.xy_unit
            self.y_vals = self.y_raw/self.xy_unit
            self.z_vals = self.z_raw/self.z_unit
        
        self.x_vals_processed = self.x_vals
        self.y_vals_processed = self.y_vals
        self.z_vals_processed = self.z_vals

    def unit_convert(
        self,
        xy_unit_target: str = 'um',
        z_unit_target: str = 'um',
    ) -> None:
        """ Convert units of the data

        Args:
          xy_unit_target: target unit for the xy coordinates
          z_unit_target: target unit for the z profile
        """
        xy_unit_target_symbol = xy_unit_target
        z_unit_target_symbol = z_unit_target
        xy_unit_target = UNIT_TABLE[xy_unit_target_symbol]
        z_unit_target = UNIT_TABLE[z_unit_target_symbol]
        
        # convert data into target units
        self.x_vals = self.x_vals*self.xy_unit/xy_unit_target
        self.y_vals = self.y_vals*self.xy_unit/xy_unit_target
        self.z_vals = self.z_vals*self.z_unit/z_unit_target

        self.x_vals_processed = (
            self.x_vals_processed*self.xy_unit/xy_unit_target)
        self.y_vals_processed = (
            self.y_vals_processed*self.xy_unit/xy_unit_target)
        self.z_vals_processed = (
            self.z_vals_processed*self.z_unit/z_unit_target)
        
        # save unit setting
        self.xy_unit_symbol = xy_unit_target_symbol
        self.z_unit_symbol = z_unit_target_symbol
        self.xy_unit = UNIT_TABLE[self.xy_unit_symbol]
        self.z_unit = UNIT_TABLE[self.z_unit_symbol]


    def plot_data(
        self, 
        dataset: str = 'processed'
    ) -> None:
        """ Plot original or processed data
        
        Args:
          dataset: choose 'original' or 'processed' data set
        """
        dataset_valid = ['original', 'processed']
        if dataset not in dataset_valid:
            raise ValueError('set_data_unit:'
                             f' dataset must be one of {dataset_valid}')
        if dataset == 'processed':
            z_vals_plot = self.z_vals_processed
            x_vals_plot = self.x_vals_processed
            y_vals_plot = self.y_vals_processed
        else:
            z_vals_plot = self.z_vals
            x_vals_plot = self.x_vals
            y_vals_plot = self.y_vals
        fig = plt.figure(figsize = (10,4))
        ax1 = fig.add_subplot(121, projection = '3d')
        im1 = ax1.plot_surface(x_vals_plot, y_vals_plot, z_vals_plot,
                               cmap=cm.PuBu)
        ax1.set_xlabel(f'x ({self.xy_unit_symbol})')
        ax1.set_ylabel(f'y ({self.xy_unit_symbol})')
        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(x_vals_plot, y_vals_plot, z_vals_plot, 
                             cmap=cm.PuBu)
        ax2.axis('equal')
        fig.colorbar(im2, ax=ax2, 
                     label = f'z ({self.z_unit_symbol})', location = 'left')
        plt.show()
    
    def level(
        self, 
        dataset: str = 'original',
        level_ref: str = 'outside_circle',
        circle_size = 1.2,
        if_plot: bool = False
    ) -> None:
        """ Level the data to the area outside a centered circle
        The leveled data will be returned to:
        self.x_vals_processed, self.y_vals_processed, self.z_vals_processed
        
        Args:
          dataset: choose 'original' or 'processed' data set to be leveled
          level_ref: choose the reference data points for leveling
          circle_size: size of the circular mask if level_ref=='outside_circle'
          if_plot: choose to plot the data selection bondaries or not
        """
        dataset_valid = ['original', 'processed']
        if dataset not in dataset_valid:
            raise ValueError('set_data_unit:'
                             f' dataset must be one of {dataset_valid}')
        level_ref_valid = ['outside_circle']
        if level_ref not in level_ref_valid:
            raise ValueError('set_data_unit:'
                             f' dataset must be one of {level_ref_valid}')

        if dataset == 'original':
            self.x_vals_processed = self.x_vals
            self.y_vals_processed = self.y_vals
            self.z_vals_processed = self.z_vals

        if level_ref == 'outside_circle':
            ref_x_row, ref_y_row, ref_z_row = self.data_outside_circle(
                circle_size = circle_size, if_plot = if_plot
            )
        
        # reconstruct data into matrix and fit the plane using least-squares
        ref_xy1 = np.c_[ref_x_row, ref_y_row, np.ones(ref_x_row.shape[0])]
        fit_plane_co, _, _, _ = scipy.linalg.lstsq(ref_xy1, ref_z_row)

        # evaluate the fitted plane on grid and substract it from the data
        z_fit_plane = (fit_plane_co[0]*self.x_vals_processed
                      +fit_plane_co[1]*self.y_vals_processed
                      +fit_plane_co[2])
        self.z_vals_processed = self.z_vals_processed - z_fit_plane

    def data_outside_circle(
        self, 
        circle_size: float = 1.2,
        if_plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Select data outside a centered circle
        
        Args:
          circle_size: ratio of the circle diameter to the width of the FOV
          if_plot: choose to plot the circle overlay with the 2D color map
        """
        # define the centered circle
        y_len, x_len = np.shape(self.z_vals_processed)
        y_span = (y_len-1)*self.y_res/self.xy_unit
        x_span = (x_len-1)*self.x_res/self.xy_unit
        circle_cy = y_span/2
        circle_cx = x_span/2
        circle_diameter = np.min([y_span, x_span])*circle_size

        # truth table for determining if each data point is outside the circle
        truthtable_outside_circle = ((self.x_vals_processed-circle_cx)**2 
                                     + (self.y_vals_processed-circle_cy)**2 
                                     >= (circle_diameter/2)**2)
        
        # get the data outside the circle
        x_outside_circle_row = self.x_vals_processed[truthtable_outside_circle]
        y_outside_circle_row = self.y_vals_processed[truthtable_outside_circle]
        z_outside_circle_row = self.z_vals_processed[truthtable_outside_circle]

        # plot the circle overlay with the 2D color map
        if if_plot == True:
            fig = plt.figure(figsize = (5,4))
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(self.x_vals_processed, self.y_vals_processed,
                               self.z_vals_processed, cmap=cm.PuBu)
            ax.axis('equal')
            circle = Circle((circle_cx, circle_cy), radius = circle_diameter/2, 
                            facecolor='None', edgecolor='r')
            ax.add_patch(circle)
            fig.colorbar(im, ax=ax, 
                        label = f'z ({self.z_unit_symbol})', location = 'left')
            plt.show()
        return x_outside_circle_row, y_outside_circle_row, z_outside_circle_row
        
    def center(self, 
               region_perc: float = 0.5,
        ) -> None:
        """ Center the data with respect to the lowest point

        The lowest point will be detected within the central region
        After centering, the asymmetric part of the data will be dropped

        Args: 
          region_perc: percentage of the region used to find minimum
          if_plot: choose if plot lines indicating center
        """
        z = self.z_vals_processed
        y_len, x_len = np.shape(z)

        # create index matrix
        y_i_row = np.arange(0, y_len)
        x_j_row = np.arange(0, x_len)
        y_i, x_j = np.meshgrid(y_i_row, x_j_row, indexing='ij')
        
        # index range of the detection region
        region_perc = region_perc
        y_i_region_len = int(y_len*region_perc)
        x_j_region_len = int(x_len*region_perc)
        
        y_i_region_start = int(y_len*(0.5-region_perc*0.5))
        x_j_region_start = int(x_len*(0.5-region_perc*0.5))

        # index and data matrix of the detection region
        y_i_region = y_i[y_i_region_start:y_i_region_start+y_i_region_len,
                         x_j_region_start:x_j_region_start+x_j_region_len]
        x_j_region = x_j[y_i_region_start:y_i_region_start+y_i_region_len,
                         x_j_region_start:x_j_region_start+x_j_region_len]
        z_region = z[y_i_region_start:y_i_region_start+y_i_region_len, 
                     x_j_region_start:x_j_region_start+x_j_region_len]

        # find the index of the center and corresponding coordinates
        center = np.unravel_index(z_region.argmin(), z_region.shape)
        y_i_center = y_i_region[center[0], center[1]]
        x_j_center = x_j_region[center[0], center[1]]
        y_center = self.y_vals_processed[y_i_center, x_j_center]
        x_center = self.x_vals_processed[y_i_center, x_j_center]

        # radius of centered data range
        z_centered_radius = np.min([(y_i_center-1)-0+1, 
                                (y_len-1)-(y_i_center+1)+1, 
                                (x_j_center-1)-0+1, 
                                (x_len-1)-(x_j_center+1)+1])
        
        # pick symmetric data around the center
        y_i_start = y_i_center-z_centered_radius
        y_i_stop = (y_i_center+1)+z_centered_radius
        x_j_start = x_j_center-z_centered_radius
        x_j_stop = (x_j_center+1)+z_centered_radius

        self.z_vals_processed = (
            self.z_vals_processed[y_i_start:y_i_stop,x_j_start:x_j_stop])
        self.x_vals_processed = (
            self.x_vals_processed[y_i_start:y_i_stop,x_j_start:x_j_stop] 
            - x_center)
        self.y_vals_processed = (
            self.y_vals_processed[y_i_start:y_i_stop,x_j_start:x_j_stop] 
            - y_center)

class FPCavity:
    """Cavity setup and mode solver

    The input takes meter as length units.
    The built-in methods use micrometer as basic unit for simulation.
    
    Attributes:


    Todos:
        accommodate a second mirror profile 
    """

    def __init__(
        self,
        mirror1: FPMirror,
        length: float,
        wavelength: float,
        waist_guess: float,
        mirror2: FPMirror = None,
        R2: float = None,
        waist_guess_range: float = 0.5,
        mode_order_combined: int = 16,
        if_window: bool = False,
        window_size: float = 1.0
    ) -> None:
        """Cavity setup

        Setup mirrors, wavelength, length of the cavity for simulation
        Units of all input arguments are meters
        
        Args:
          mirror1: first mirror profile in the FP cavity
          mirror2: second mirror profile in the FP cavity
          currently only supports None, will add the functionalities in the
          future
          R2: radius of curvature of mirror2, only needed when mirror2 == None
          length: cavity length
          wavelength: optical wavelength for simulation
          waist_guess: initial guess for waist size
          waist_guess_range: percentage of the waist guess range with respect
          to waist_guess
          mode_order_combined: highest combined mode order used for HG solver 
          if_window: choose to use window function or not for simulation
          for a full mirror profile, the window function is not required
          window_size: parameter that controls the window size if the window is turned on
        """
        self.mirror1 = mirror1
        self.length = length
        self.wavelength = wavelength

        self.mirror2 = mirror2

        self.waist_guess = waist_guess
        self.wasit_guess_range = waist_guess_range
        self.mode_order_combined = mode_order_combined

        self.base_unit_symbol = 'um'
        self.base_unit = UNIT_TABLE[self.base_unit_symbol]

        # convert cavity parameters into base units
        u = self.base_unit
        self.length_u = length/u
        self.wavelength_u = wavelength/u
        self.k_u = 2*np.pi/self.wavelength_u
        self.waist_guess_u = waist_guess/u
        self.speed_of_light_u = SPEED_OF_LIGHT/u
        self.FSR = self.speed_of_light_u/(2*self.length_u)

        # set up two mirrors
        self.mirror1.unit_convert(xy_unit_target = 'um', z_unit_target = 'um')
        if self.mirror2 == None:
            mirror2_shape = np.shape(self.mirror1.z_vals_processed)
            mirror2_y = self.mirror1.y_vals_processed
            mirror2_x = self.mirror1.x_vals_processed
            if R2 == None:
                mirror2_z = np.zeros(mirror2_shape)
            else:
                self.R2 = R2
                self.R2_u = R2/u
                mirror2_z = (mirror2_y**2+mirror2_x**2)/(2*self.R2_u)
            self.mirror2 = FPMirror(
                data_x = mirror2_x, data_y = mirror2_y, data_z = mirror2_z,
                xy_unit = 'um', z_unit = 'um')
            #save the xy resolution
            self.xy_res_u = self.mirror1.xy_res/u
        else:
            pass
            # Todo: check if simulation range/grids match
            # It gets tricky if the second mirror is actually provided

        # set up simulation window
        self.if_window = if_window
        if self.if_window == True:
            self.window_size = window_size
            self.sim_window = self.window(size = window_size)
        else:
            self.sim_window = np.ones(np.shape(self.mirror1.z_vals_processed))
    
    def window(
        self,
        size: float = 1.0
    )->np.ndarray:
        """Simulation window function

        This window function is needed for the cases where the FOV of the 
        mirror profile measurement is limited

        Args:
          size: size of the window, from 0.0 to 1.5, and 1.5 corresponds to
          taking the full data
        """
        y_len, x_len = np.shape(self.mirror1.z_vals_processed)
        a = 50
        x0 = a*size
        abs_y = np.linspace(-1*a,a,y_len)
        abs_x = np.linspace(-1*a,a,x_len)
        ABS_Y, ABS_X = np.meshgrid(abs_y, abs_x, indexing='ij')
        ABS_Z = (0.5*(1+erf(np.sqrt(ABS_X**2+ABS_Y**2)+x0))
                 *0.5*(1-erf(np.sqrt(ABS_X**2+ABS_Y**2)-x0)))
        return ABS_Z

    def waist_search1(
        self,
        if_print: bool = False
    ) -> None:
        """Beam waist search function for the plano-concave cavity

        This function does one round of beam propagation and search for the 
        optimal beam waist that has minimal change after the propagation

        Returns:
          waist_opt: optimized beam waist size
          if_print: choose to print the optimized result or not
        
        Warnings:
          Potential insufficient range: please try increasing the waist_guess_range
        """
        w0_guess = self.waist_guess_u
        w0_range = self.wasit_guess_range
        waist_opt = solver_core.waist_search1(
            X = self.mirror1.x_vals_processed, Y = self.mirror1.y_vals_processed,
            xy_res = self.xy_res_u, Z = self.mirror1.z_vals_processed,
            k = self.k_u, L = self.length_u,
            w0_guess = w0_guess, w0_range = w0_range,
            sim_window = self.sim_window)
        self.waist_u = waist_opt
        if (waist_opt == w0_guess*(1-w0_range) 
            or waist_opt == w0_guess*(1+w0_range)):
            warnings.warn('Optimized beam waist reaches the optimization boundary!')
        if if_print == True:
            print('w0_init is %.2f um, optimized w0 is %.2f um.'%(w0_guess, waist_opt))
            zR = np.pi*waist_opt**2/self.wavelength_u
            wL = waist_opt*np.sqrt(1+self.length_u**2/zR**2)
            R = self.length_u+zR**2/self.length_u
            print(f'Effective ROC is zR is %.3f mm, wL is %.2f um.'%(R/(1e3), wL))
    
    def mode_solve(
        self
    ) -> None:
        """Solve the eigen modes of the cavity

        This function constructs the scattering matrix and solve the eigen
        problem of it.
        
        Add the following attributes to the class:
          mode_num: total number of the Hermite-Gaussian modes used in the solver
          eigen_value: the eigenvalues of the problem
          eigen_vector: the eigenvectors of the problem
          loss: losses of the modes
          freq: frequencies of the modes
          mode: the corresponding modes in xy-basis
        """
        mode_num, eigen_value, eigen_vector, loss, mode = solver_core.scatter_eigen(
            X = self.mirror1.x_vals_processed, Y = self.mirror1.y_vals_processed,
            xy_res = self.xy_res_u,
            mirror1_Z = self.mirror1.z_vals_processed,
            mirror2_Z = self.mirror2.z_vals_processed,
            k = self.k_u, L = self.length_u,
            w0 = self.waist_u, mode_order_combined = self.mode_order_combined,
            sim_window = self.sim_window
        )
        self.mode_num = mode_num
        self.eigen_value = eigen_value
        self.eigen_vector = eigen_vector
        
        self.loss = loss
        eigen_frequency = (np.angle(eigen_value)-0)/(2*np.pi)*self.FSR
        for m in np.arange(len(eigen_frequency)):
            if eigen_frequency[m] < 0:
                eigen_frequency[m] += self.FSR
        self.frequency = eigen_frequency
        self.mode = mode

    def mode_sort(
        self,
        ref: str = 'mode',
        HG_mode_order: int = 0
    ) -> None:
        """Sort the modes

        Args:
          ref: choose to sort the modes according to 'loss', 'mode' composition
          or 'frequency'
          HG_mode_order: specify which order of the Hermite-Gaussian mode is 
          referenced to if 'mode' is chosen 
        """
        ref_valid = ['mode', 'loss', 'frequency']
        if ref not in ref_valid:
            raise ValueError(f'ref: ref must be one of {ref_valid}')
        if ref == 'mode':
            mode_comp = abs(self.eigen_vector[HG_mode_order])**2
            sort_ind = np.argsort(mode_comp)[::-1]
        elif ref == 'loss':
            sort_ind = np.argsort(self.loss)
        else:
            sort_ind = np.argsort(self.frequency)
        
        self.eigen_value = self.eigen_value[sort_ind]
        self.eigen_vector = self.eigen_vector[:, sort_ind]
        self.loss = self.loss[sort_ind]
        self.frequency = self.frequency[sort_ind]
        self.mode = self.mode[sort_ind, :, :]

    def mode_plot(
        self,
        mode_order: int = 0
    ) -> None:
        """Plot the specified mode
        
        Args:
          mode_order: order of the specified mode in the currently saved list 
        """
        fig = plt.figure(figsize = (10,4))
        ax1 = fig.add_subplot(121)
        im1 = ax1.contourf(
            self.mirror1.x_vals_processed, self.mirror1.y_vals_processed, 
            abs(self.mode[mode_order, :, :])**2, cmap=cm.hot
        )
        ax1.set_title(f'{mode_order}th-Order Mode', fontsize = 14)
        ax1.set_xlabel(f'x ({self.base_unit_symbol})')
        ax1.set_ylabel(f'y ({self.base_unit_symbol})')
        ax1.axis('equal')
        # fig.colorbar(im1, ax = ax1)
        ax2 = fig.add_subplot(122)
        ax2.bar(range(self.mode_num), abs(self.eigen_vector[:, mode_order])**2)
        ax2.set_title('Loss = %.1f ppm'%(self.loss[mode_order]*1e6), fontsize=14)
        ax2.set_xlabel('Hermite-Gaussian mode order', fontsize=14)
        ax2.set_ylabel('Energy composition', fontsize=14)
        ax2.set_yscale('log')
        ax2.set_ylim(ymin = 1e-3, ymax = 1e0)
        ax2.axhline(y=0.01, color='k',linestyle='-.')
        plt.show()


if __name__ == '__main__':
    # set up an instance of the FPMirror class with the profile data for processing
    mirror1 = FPMirror('example_profile.datx')
    # level the mirror profile with respect to the outer area
    mirror1.level()
    # center the mirror profile with respect to the deepest point
    mirror1.center()
    # visualize the mirror profile
    mirror1.plot_data()
    
    # setup an instance of the FPCavity class with the mirror and cavity settings
    cavity1 = FPCavity(
        mirror1 = mirror1, 
        length = 170.80e-6,
        wavelength = 854e-9, 
        waist_guess = 7e-6,
        mode_order_combined = 18,
        if_window = True,
        window_size = 0.9
    )
    # search for the wasit of the cavity
    cavity1.waist_search1(if_print = True)
    # solve the eigenmodes of the cavity
    cavity1.mode_solve()
    # sort the eigenmodes according to the fundamental mode composition
    cavity1.mode_sort(ref = 'mode', HG_mode_order = 0)
    # plot the result
    cavity1.mode_plot(mode_order = 0)
