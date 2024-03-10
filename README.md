# fp-mode-solver
Eigen-mode solver for optical Fabry–Pérot resonators with standard/non-standard mirror shapes.

![example_mirror](https://github.com/njjingogogo/fp-mode-solver/assets/162845168/e186a5ed-6f33-4017-87cf-7864d418a05b)
Given an input mirror profile like above, this mode solver can calculate the eigenmodes of the plano-concave cavity built with this mirror and corresponding losses shown below.
![example_eigenmode](https://github.com/njjingogogo/fp-mode-solver/assets/162845168/f867f9e7-f37d-4e7b-a5af-ba5e7f94dc6c)

# How to use
FPsolver.py wraps the main classes used in this solver - FPMirror & FPCavity.

To see an example, run FPsolver.py. Here are detailed explanations to the main function

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

# Reference
The core algorithm for the solver is published in the following papers

[1] [Numerically Accelerated Development Cycle for Ultra-high Finesse Micro-fabricated Resonators](https://arxiv.org/abs/1502.01532)

[2] [Transverse-mode coupling and diffraction loss in tunable Fabry-Pérot microcavities](https://arxiv.org/abs/1502.01532)
