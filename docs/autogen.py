import glob
import os
import shutil

from keras_autodoc import DocumentationGenerator

pages = {
    "fit.md": [
        "burstfit.fit.BurstFit",
        "burstfit.fit.BurstFit.validate",
        "burstfit.fit.BurstFit.precalc",
        "burstfit.fit.BurstFit.make_spectra",
        "burstfit.fit.BurstFit.fitcycle",
        "burstfit.fit.BurstFit.initial_profilefit",
        "burstfit.fit.BurstFit.initial_spectrafit",
        "burstfit.fit.BurstFit.sgram_fit",
        "burstfit.fit.BurstFit.fit_all_components",
        "burstfit.fit.BurstFit.fitall",
        "burstfit.fit.BurstFit.run_mcmc",
        "burstfit.fit.BurstFit.get_off_pulse_region",
        "burstfit.fit.BurstFit.calc_redchisq",
        "burstfit.fit.BurstFit.model_from_params",
        "burstfit.fit.BurstFit.get_physical_parameters",
    ],
    "data.md": [
        "burstfit.data.BurstData",
        "burstfit.data.BurstData.prepare_data",
        "burstfit.data.BurstData.prepare_input_mask",
        "burstfit.data.BurstData.mask_channels",
        "burstfit.data.BurstData.normalise_data",
        "burstfit.data.BurstData.crop_dedispersed_data",
    ],
    "io.md":[
        "burstfit.io.BurstIO",
        "burstfit.io.BurstIO.set_attributes_to_save",
        "burstfit.io.BurstIO.save_results",
        "burstfit.io.BurstIO.read_json_and_precalc",
        "burstfit.io.BurstIO.set_metadata",
        "burstfit.io.BurstIO.set_classes_from_dict",
    ],
    "mcmc.md":[
        "burstfit.mcmc.MCMC",
        "burstfit.mcmc.MCMC.lnprior",
        "burstfit.mcmc.MCMC.lnprob",
        "burstfit.mcmc.MCMC.lnlk",
        "burstfit.mcmc.MCMC.set_initial_pos",
        "burstfit.mcmc.MCMC.set_priors",
        "burstfit.mcmc.MCMC.run_mcmc",
        "burstfit.mcmc.MCMC.get_chain",
        "burstfit.mcmc.MCMC.print_results",
        "burstfit.mcmc.MCMC.plot",
        "burstfit.mcmc.MCMC.make_autocorr_plot",
    ],
    "model.md":[
        "burstfit.model.Model",
        "burstfit.model.Model.evaluate",
        "burstfit.model.Model.get_param_dict",
        "burstfit.model.SgramModel",
        "burstfit.model.SgramModel.evaluate",
    ],
    "utils/astro.md":[
        "burstfit.utils.astro.dedisperse",
        "burstfit.utils.astro.finer_dispersion_correction",
        "burstfit.utils.astro.radiometer",
    ],
    "utils/functions.md":[
        "burstfit.utils.functions.gauss",
        "burstfit.utils.functions.gauss_norm",
        "burstfit.utils.functions.gauss_norm2",
        "burstfit.utils.functions.gauss_norm3",
        "burstfit.utils.functions.pulse_fn",
        "burstfit.utils.functions.pulse_fn_vec",
        "burstfit.utils.functions.sgram_fn",
        "burstfit.utils.functions.sgram_fn_vec"
    ],
    "utils/math.md":[
        "burstfit.utils.math.f_test",
        "burstfit.utils.math.tests",
        "burstfit.utils.math.fmae",
        "burstfit.utils.math.transform_parameters",
    ],
    "utils/plotter.md":[
        "burstfit.utils.plotter.plot_1d_fit",
        "burstfit.utils.plotter.plot_2d_fit",
        "burstfit.utils.plotter.plot_fit_results",
        "burstfit.utils.plotter.plot_me",
        "burstfit.utils.plotter.plot_mcmc_results",
        "burstfit.utils.plotter.autocorr_plot",
    ],
}

# Generate documentation from the installed package
doc_generator = DocumentationGenerator(
    pages, "https://github.com/thepetabyteproject/burstfit/blob/main"
)
doc_generator.generate("./sources")

# Make readme as the start page
shutil.copyfile("../README.md", "sources/index.md")
#shutil.copyfile("../CODE_OF_CONDUCT.md", "sources/CODE_OF_CONDUCT.md")
#shutil.copyfile("../CONTRIBUTING.md", "sources/CONTRIBUTING.md")

# Make the dir for tutorials
os.mkdir("sources/ipynb")
for nb in glob.glob("../examples/*ipynb"):
    file_name = os.path.basename(nb)
    os.symlink(os.path.abspath(nb), "sources/ipynb/" + file_name)

# Convert all note tabs so that it looks cooler with the material theme
linebreaker_list = ["Args:", "Examples:", "Returns:", "Attributes:", "Raises:"]

for dname, dirs, files in os.walk("sources"):
    for fname in files:
        fpath = os.path.join(dname, fname)
        with open(fpath) as f:
            s = f.read()
        s = s.replace("Note:", "!!! note")
        s = s.replace("**Note**:", "!!! note")
        for string in linebreaker_list:
            s = s.replace(string, string + " \n")
        with open(fpath, "w") as f:
            f.write(s)
