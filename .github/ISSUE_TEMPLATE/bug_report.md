---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

Known issues. Do not open issue, if you faced with problem from the list:

> ImportError: cannot import name 'ui_toprow' from 'modules' (unknown location)

You need to update automatic1111 webui. Launch `git pull` command inside webui root

> ModuleNotFoundError: No module named 'scripts.sam'

You need to install https://github.com/continue-revolution/sd-webui-segment-anything
Do not confuse with Inpaint Anything!

> sam = sam_model_registrymodel_type
> KeyError: ''

Segment Anything extension currently doesn't support FastSam and Matting-Anything. Ask about this here: https://github.com/continue-revolution/sd-webui-segment-anything/issues/135
