#!/bin/sh

jupyter nbconvert *.ipynb --to slides --TemplateExporter.exclude_input=True
# Include slide numbering:
sed -i 's/\(transition: "slide",\)/\1 slideNumber: "c\/t",/' *.slides.html

