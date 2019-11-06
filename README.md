# Topic Modeling on the Crowd RE Dataset

## Build

To build the pdf run `latexmk main`

## Contribute

Content should be grouped into separate files and added to the *content* directory appropriately. The individual files can then be included in the main.tex using the `\input{contet/...}` directive.

Any acronyms introduced to the document, may be added to the *misc/acronyms.tex* in alphabetical order.

Bibliographic references go to the *references.bib* and can further be referenced using the `\cite{...}` or the `\cite[p123]{...}` command.


## LaTeX Tools

When using the LaTeXTools Plugin for Sublime Text, make sure to configure your *.sublime-project* as follows:

```json
{
	...
	"settings" : {
        "TEXroot": "main.tex",
        "tex_file_exts": [".tex", ".tikz"],
        "output_directory": "build",
        "copy_output_on_build": true
    }
}
```