        # tool for loading the range from above.


        load_range_from_above = Button(
            description="Click to load projection range from plot tab.",
            disabled=True,
            button_style="info",
            tooltip="Make sure to choose all of the buttons above before clicking this button",
            icon="",
            layout=Layout(width="95%", justify_content="center"),
        )
        load_range_from_above.on_click(load_range_from_above_onclick)


def radio_align_full_partial(change):
            if change.new == 1:
                projection_range_x_alignment.disabled = False
                projection_range_y_alignment.disabled = False
                load_range_from_above.disabled = False
                self.metadata["aligndata"] = True
                load_range_from_above.description = (
                    "Click to load projection range from above."
                )
                load_range_from_above.icon = ""
            elif change.new == 0:
                if "range_y_link" in locals() or "range_y_link" in globals():
                    range_y_link.unlink()
                    range_x_link.unlink()
                    load_range_from_above.button_style = "info"
                    load_range_from_above.description = (
                        "Unlinked ranges. Enable partial range to link again."
                    )
                    load_range_from_above.icon = "unlink"
                projection_range_x_alignment.value = [0, tomo.prj_imgs.shape[2] - 1]
                projection_range_x_alignment.disabled = True
                projection_range_y_alignment.value = [0, tomo.prj_imgs.shape[1] - 1]
                projection_range_y_alignment.disabled = True
                load_range_from_above.disabled = True
                self.metadata["aligndata"] = False


        ############################# METHOD CHOOSER GRID ############################
        grid_alignment = GridspecLayout(2, 3)
        # align_FBP_CUDA = Checkbox(description="FBP_CUDA")
        # align_FBP_CUDA_option1 = Checkbox(description="option1", disabled=True)
        # align_FBP_CUDA_option2 = Checkbox(description="option2", disabled=True)
        # align_FBP_CUDA_option3 = Checkbox(description="option3", disabled=True)
        # align_FBP_CUDA_option_list = [
        #     align_FBP_CUDA_option1,
        #     align_FBP_CUDA_option2,
        #     align_FBP_CUDA_option3,
        # ]

        align_SIRT_CUDA = Checkbox(description="SIRT_CUDA", value=1)
        align_SIRT_CUDA_option1 = Checkbox(description="Faster", disabled=False)
        align_SIRT_CUDA_option2 = Checkbox(description="Fastest", disabled=False)
        align_SIRT_CUDA_option3 = Checkbox(description="option3", disabled=False)
        align_SIRT_CUDA_option_list = [align_SIRT_CUDA_option1, align_SIRT_CUDA_option2]

        # align_SART_CUDA = Checkbox(description="SART_CUDA")
        # align_SART_CUDA_option1 = Checkbox(description="option1", disabled=True)
        # align_SART_CUDA_option2 = Checkbox(description="option2", disabled=True)
        # align_SART_CUDA_option3 = Checkbox(description="option3", disabled=True)
        # align_SART_CUDA_option_list = [
        #     align_SART_CUDA_option1,
        #     align_SART_CUDA_option2,
        #     align_SART_CUDA_option3,
        # ]

        # align_CGLS_CUDA = Checkbox(description="CGLS_CUDA")
        # align_CGLS_CUDA_option1 = Checkbox(description="option1", disabled=True)
        # align_CGLS_CUDA_option2 = Checkbox(description="option2", disabled=True)
        # align_CGLS_CUDA_option3 = Checkbox(description="option3", disabled=True)
        # align_CGLS_CUDA_option_list = [
        #     align_CGLS_CUDA_option1,
        #     align_CGLS_CUDA_option2,
        #     align_CGLS_CUDA_option3,
        # ]

        # align_MLEM_CUDA = Checkbox(description="MLEM_CUDA")
        # align_MLEM_CUDA_option1 = Checkbox(description="option1", disabled=True)
        # align_MLEM_CUDA_option2 = Checkbox(description="option2", disabled=True)
        # align_MLEM_CUDA_option3 = Checkbox(description="option3", disabled=True)
        # align_MLEM_CUDA_option_list = [
        #     align_MLEM_CUDA_option1,
        #     align_MLEM_CUDA_option2,
        #     align_MLEM_CUDA_option3,
        # ]

        align_method_list = [
            # align_FBP_CUDA,
            align_SIRT_CUDA,
            # align_SART_CUDA,
            # align_CGLS_CUDA,
            # align_MLEM_CUDA,
        ]

        def toggle_on(change, opt_list, dictname):
            if change.new == 1:
                self.metadata["methods"][dictname] = {}
                for option in opt_list:
                    option.disabled = False
            if change.new == 0:
                self.metadata["methods"].pop(dictname)
                for option in opt_list:
                    option.value = 0
                    option.disabled = True

        # align_FBP_CUDA.observe(
        #     functools.partial(
        #         toggle_on, opt_list=align_FBP_CUDA_option_list, dictname="FBP_CUDA"
        #     ),
        #     names=["value"],
        # )
        align_SIRT_CUDA.observe(
            functools.partial(
                toggle_on, opt_list=align_SIRT_CUDA_option_list, dictname="SIRT_CUDA"
            ),
            names=["value"],
        )
        # align_SART_CUDA.observe(
        #     functools.partial(
        #         toggle_on, opt_list=align_SART_CUDA_option_list, dictname="SART_CUDA"
        #     ),
        #     names=["value"],
        # )
        # align_CGLS_CUDA.observe(
        #     functools.partial(
        #         toggle_on, opt_list=align_CGLS_CUDA_option_list, dictname="CGLS_CUDA"
        #     ),
        #     names=["value"],
        # )
        # align_MLEM_CUDA.observe(
        #     functools.partial(
        #         toggle_on, opt_list=align_MLEM_CUDA_option_list, dictname="MLEM_CUDA"
        #     ),
        #     names=["value"],
        # )

        def create_option_dictionary(opt_list):
            opt_dictionary = {opt.description: opt.value for opt in opt_list}
            return opt_dictionary

        def create_dict_on_checkmark(change, opt_list, dictname):
            self.metadata["methods"][dictname] = create_option_dictionary(opt_list)

        # Makes generator for mapping of options to observe functions.

        # list(
        #     (
        #         opt.observe(
        #             functools.partial(
        #                 create_dict_on_checkmark,
        #                 opt_list=align_FBP_CUDA_option_list,
        #                 dictname="FBP_CUDA",
        #             ),
        #             names=["value"],
        #         )
        #         for opt in align_FBP_CUDA_option_list
        #     )
        # )
        list(
            (
                opt.observe(
                    functools.partial(
                        create_dict_on_checkmark,
                        opt_list=align_SIRT_CUDA_option_list,
                        dictname="SIRT_CUDA",
                    ),
                    names=["value"],
                )
                for opt in align_SIRT_CUDA_option_list
            )
        )
        # list(
        #     (
        #         opt.observe(
        #             functools.partial(
        #                 create_dict_on_checkmark,
        #                 opt_list=align_SART_CUDA_option_list,
        #                 dictname="SART_CUDA",
        #             ),
        #             names=["value"],
        #         )
        #         for opt in align_SART_CUDA_option_list
        #     )
        # )
        # list(
        #     (
        #         opt.observe(
        #             functools.partial(
        #                 create_dict_on_checkmark,
        #                 opt_list=align_CGLS_CUDA_option_list,
        #                 dictname="CGLS_CUDA",
        #             ),
        #             names=["value"],
        #         )
        #         for opt in align_CGLS_CUDA_option_list
        #     )
        # )
        # list(
        #     (
        #         opt.observe(
        #             functools.partial(
        #                 create_dict_on_checkmark,
        #                 opt_list=align_MLEM_CUDA_option_list,
        #                 dictname="MLEM_CUDA",
        #             ),
        #             names=["value"],
        #         )
        #         for opt in align_MLEM_CUDA_option_list
        #     )
        # )

        def fill_grid(method, opt_list, linenumber, grid):
            grid[linenumber, 0] = method
            i = 1
            for option in opt_list:
                grid[linenumber, i] = option
                i += 1

        # fill_grid(align_FBP_CUDA, align_FBP_CUDA_option_list, 1, grid_alignment)
        fill_grid(align_SIRT_CUDA, align_SIRT_CUDA_option_list, 1, grid_alignment)
        # fill_grid(align_SART_CUDA, align_SART_CUDA_option_list, 3, grid_alignment)
        # fill_grid(align_CGLS_CUDA, align_CGLS_CUDA_option_list, 4, grid_alignment)
        # fill_grid(align_MLEM_CUDA, align_MLEM_CUDA_option_list, 5, grid_alignment)

        grid_column_headers = ["Method", "Option 1", "Option 2"]
        for i, method in enumerate(grid_column_headers):
            grid_alignment[0, i] = Label(
                value=grid_column_headers[i], layout=Layout(justify_content="center")
            )