class UploaderBase(ABC):
    def __init__(self):

        self.quick_path_search = Textarea(
            placeholder=r"Z:\swelborn",
            style=extend_description_style,
            disabled=False,
            layout=Layout(align_items="stretch"),
        )

        self.quick_path_label = Label("Quick path search")
        self.filepath = None
        self.filename = None
        self.filechooser = FileChooser()
        self.import_button = Button(
            icon="upload",
            style={"font_size": "35px"},
            button_style="",
            layout=Layout(width="75px", height="86px"),
            disabled=True,
        )

    @abstractmethod
    def update_filechooser_from_quicksearch(self, change):
        ...

    @abstractmethod
    def update_quicksearch_from_filechooser(self):
        ...


class PrenormUploader(UploaderBase):
    def __init__(self, PrenormProjections, Import):
        super().__init__()
        self.projections = PrenormProjections
        self.Import = Import
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Prenormalized Data"

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filepath = path
        self.filechooser.reset(path=path)

        self.import_button.button_style = "info"
        self.import_button.disabled = False

    def update_quicksearch_from_filechooser(self):

        self.filepath = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filepath)


class RawUploader_SSRL62(UploaderBase):
    def __init__(self, RawProjections, Import):
        super().__init__()
        self.projections = RawProjections
        self.Import = Import
        self.quick_path_search.observe(
            self.update_filechooser_from_quicksearch, names="value"
        )
        self.filechooser.register_callback(self.update_quicksearch_from_filechooser)
        self.filechooser.title = "Import Raw XRM Folder"

    def update_filechooser_from_quicksearch(self, change):
        path = pathlib.Path(change.new)
        self.filepath = path
        self.filechooser.reset(path=path)
        textfiles = self.projections._file_finder(path, [".txt"])
        if textfiles == []:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have any .txt files, please try another one."
                )
            return

        scan_info_filepath = (
            path / [file for file in textfiles if "ScanInfo" in file][0]
        )
        if scan_info_filepath != []:
            self.projections.import_metadata(path)
            self.metadata_table = self.projections.metadata_to_DataFrame()
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
        else:
            with self.metadata_table_output:
                self.metadata_table_output.clear_output(wait=True)
                print(
                    "This folder doesn't have a ScanInfo file, please try another one."
                )

    def update_quicksearch_from_filechooser(self):

        self.filepath = pathlib.Path(self.filechooser.selected_path)
        self.filename = self.filechooser.selected_filename
        self.quick_path_search.value = str(self.filepath)
        # metadata must be set here in case tomodata is created (for folder
        # import). this can be changed later.
        self.projections.import_metadata(self.filepath)
        self.metadata_table = self.projections.metadata_to_DataFrame()

        with self.metadata_table_output:
            self.metadata_table_output.clear_output(wait=True)
            display(self.metadata_table)
            self.import_button.button_style = "info"
            self.import_button.disabled = False
