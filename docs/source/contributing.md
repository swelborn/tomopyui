# Contributing

Got a feature you would like to see added? Found a bug? This is a v0.0.1 project without a set plan (other than to help you out with your tomography data). You can report an issue by opening one at the [issues](https://github.com/samwelborn/tomopyui/issues) page. You can suggest an edit to the documentation you are reading right now. You can request a new feature, or even make one yourself and submit a pull request.

## Code Improvements

All development for this app happens on GitHub [here](https://github.com/samwelborn/tomopyui). You should work with a [conda](https://www.anaconda.com/products/individual) environment.

You should follow the instructions on the {doc}`install` page to find out how you can help develop this code. Once you do find out how: make some changes, see if they work, and and tell us about it [here](https://github.com/samwelborn/tomopyui).

### Seeing your changes

If you are working in a Jupyter Notebook, then in order to see your code changes you will need to either:

- Restart the Kernel every time you make a change to the code.
- Make the function reload from the source file every time you run it by using [autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html), e.g.:

  ```python
  %load_ext autoreload
  %autoreload 2
  %matplotlib ipympl
  import tomopyui.widgets.main as main

  dashboard_output, dashboard, file_import, prep, center, align, recon, dataexplorer = main.create_dashboard("APS") # can be "SSRL_62C", "ALS_832", "APS"
  dashboard_output
  ```

### Working with Git

Using Git/GitHub can be [confusing](https://xkcd.com/1597), so if you're new to Git, you may find it helpful to use a program like [GitHub Desktop](https://desktop.github.com) and to follow a [guide](https://github.com/firstcontributions/first-contributions#first-contributions).

Also feel free to ask for help/advice on the relevant GitHub [issue](https://github.com/samwelborn/tomopyui/issues).
