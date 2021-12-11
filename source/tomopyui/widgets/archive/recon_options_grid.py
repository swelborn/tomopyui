from ipywidgets import *
import functools

recon_grid = GridspecLayout(7, 4)

FP_CUDA = Checkbox(description="FP_CUDA")
FP_CUDA_option1 = Checkbox(description="option1", disabled=True)
FP_CUDA_option2 = Checkbox(description="option2", disabled=True)
FP_CUDA_option3 = Checkbox(description="option3", disabled=True)
FP_CUDA_option_list = [FP_CUDA_option1, FP_CUDA_option2, FP_CUDA_option3]

BP_CUDA = Checkbox(description="BP_CUDA")
BP_CUDA_option1 = Checkbox(description="option1", disabled=True)
BP_CUDA_option2 = Checkbox(description="option2", disabled=True)
BP_CUDA_option3 = Checkbox(description="option3", disabled=True)
BP_CUDA_option_list = [BP_CUDA_option1, BP_CUDA_option2, BP_CUDA_option3]

FBP_CUDA = Checkbox(description="FBP_CUDA")
FBP_CUDA_option1 = Checkbox(description="option1", disabled=True)
FBP_CUDA_option2 = Checkbox(description="option2", disabled=True)
FBP_CUDA_option3 = Checkbox(description="option3", disabled=True)
FBP_CUDA_option_list = [FBP_CUDA_option1, FBP_CUDA_option2, FBP_CUDA_option3]

SIRT_CUDA = Checkbox(description="SIRT_CUDA")
SIRT_CUDA_option1 = Checkbox(description="option1", disabled=True)
SIRT_CUDA_option2 = Checkbox(description="option2", disabled=True)
SIRT_CUDA_option3 = Checkbox(description="option3", disabled=True)
SIRT_CUDA_option_list = [SIRT_CUDA_option1, SIRT_CUDA_option2, SIRT_CUDA_option3]

SART_CUDA = Checkbox(description="SART_CUDA")
SART_CUDA_option1 = Checkbox(description="option1", disabled=True)
SART_CUDA_option2 = Checkbox(description="option2", disabled=True)
SART_CUDA_option3 = Checkbox(description="option3", disabled=True)
SART_CUDA_option_list = [SART_CUDA_option1, SART_CUDA_option2, SART_CUDA_option3]
CGLS_CUDA = Checkbox(description="CGLS_CUDA")
CGLS_CUDA_option1 = Checkbox(description="option1", disabled=True)
CGLS_CUDA_option2 = Checkbox(description="option2", disabled=True)
CGLS_CUDA_option3 = Checkbox(description="option3", disabled=True)
CGLS_CUDA_option_list = [CGLS_CUDA_option1, CGLS_CUDA_option2, CGLS_CUDA_option3]

MLEM_CUDA = Checkbox(description="MLEM_CUDA")
MLEM_CUDA_option1 = Checkbox(description="option1", disabled=True)
MLEM_CUDA_option2 = Checkbox(description="option2", disabled=True)
MLEM_CUDA_option3 = Checkbox(description="option3", disabled=True)
MLEM_CUDA_option_list = [MLEM_CUDA_option1, MLEM_CUDA_option2, MLEM_CUDA_option3]

method_list = [FP_CUDA, BP_CUDA, FBP_CUDA, SIRT_CUDA, SART_CUDA, CGLS_CUDA, MLEM_CUDA]


def toggle_on(change, opt_list, dictname):
    if change.new == 1:
        reconmetadata[dictname] = {}
        for option in opt_list:
            option.disabled = False
    if change.new == 0:
        reconmetadata.pop(dictname)
        for option in opt_list:
            option.value = 0
            option.disabled = True


FP_CUDA.observe(
    functools.partial(toggle_on, opt_list=FP_CUDA_option_list, dictname="FP_CUDA"),
    names=["value"],
)
BP_CUDA.observe(
    functools.partial(toggle_on, opt_list=BP_CUDA_option_list, dictname="BP_CUDA"),
    names=["value"],
)
FBP_CUDA.observe(
    functools.partial(toggle_on, opt_list=FBP_CUDA_option_list, dictname="FBP_CUDA"),
    names=["value"],
)
SIRT_CUDA.observe(
    functools.partial(toggle_on, opt_list=SIRT_CUDA_option_list, dictname="SIRT_CUDA"),
    names=["value"],
)
SART_CUDA.observe(
    functools.partial(toggle_on, opt_list=SART_CUDA_option_list, dictname="SART_CUDA"),
    names=["value"],
)
CGLS_CUDA.observe(
    functools.partial(toggle_on, opt_list=CGLS_CUDA_option_list, dictname="CGLS_CUDA"),
    names=["value"],
)
MLEM_CUDA.observe(
    functools.partial(toggle_on, opt_list=MLEM_CUDA_option_list, dictname="MLEM_CUDA"),
    names=["value"],
)


def create_option_dictionary(opt_list):
    opt_dictionary = {opt.description: opt.value for opt in opt_list}
    return opt_dictionary


def create_dict_on_checkmark(change, opt_list, dictname):
    reconmetadata[dictname] = create_option_dictionary(opt_list)


# Makes generator for mapping of options to observe functions. Allows for the check boxes to be clicked, and sends the results to
# the reconmetadata dictionary.
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=FP_CUDA_option_list,
                dictname="FP_CUDA",
            ),
            names=["value"],
        )
        for opt in FP_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=BP_CUDA_option_list,
                dictname="BP_CUDA",
            ),
            names=["value"],
        )
        for opt in BP_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=FBP_CUDA_option_list,
                dictname="FBP_CUDA",
            ),
            names=["value"],
        )
        for opt in FBP_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=SIRT_CUDA_option_list,
                dictname="SIRT_CUDA",
            ),
            names=["value"],
        )
        for opt in SIRT_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=SART_CUDA_option_list,
                dictname="SART_CUDA",
            ),
            names=["value"],
        )
        for opt in SART_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=CGLS_CUDA_option_list,
                dictname="CGLS_CUDA",
            ),
            names=["value"],
        )
        for opt in CGLS_CUDA_option_list
    )
)
list(
    (
        opt.observe(
            functools.partial(
                create_dict_on_checkmark,
                opt_list=MLEM_CUDA_option_list,
                dictname="MLEM_CUDA",
            ),
            names=["value"],
        )
        for opt in MLEM_CUDA_option_list
    )
)


def fill_grid(method, opt_list, linenumber, grid):
    grid[linenumber, 0] = method
    i = 1
    for option in opt_list:
        grid[linenumber, i] = option
        i += 1


fill_grid(FP_CUDA, FP_CUDA_option_list, 0, recon_grid)
fill_grid(BP_CUDA, BP_CUDA_option_list, 1, recon_grid)
fill_grid(FBP_CUDA, FBP_CUDA_option_list, 2, recon_grid)
fill_grid(SIRT_CUDA, SIRT_CUDA_option_list, 3, recon_grid)
fill_grid(SART_CUDA, SART_CUDA_option_list, 4, recon_grid)
fill_grid(CGLS_CUDA, CGLS_CUDA_option_list, 5, recon_grid)
fill_grid(MLEM_CUDA, MLEM_CUDA_option_list, 6, recon_grid)

# FP_CUDA.observe(functools.partial(toggle_on, opt_list=FP_CUDA_option_list), names=['value'])

return recon_grid