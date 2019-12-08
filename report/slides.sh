#!/bin/sh

jupyter nbconvert *.ipynb --to slides --TemplateExporter.exclude_input=True

