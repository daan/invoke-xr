# invoke-XR

## Install
```
cd xr-invoke
uv init
```

## prepare

get access to shapeNetSem and make a symlink "ShapenetSem" in this archive
you will need ShapenetSem/metadata.csv and ShapenetSem/models-COLLADA/

then use the convert_select_categories noebook to convert/scale the collada models to glb and produce a csv.

run ollama with llama3.2:3b

## run
```
python server.py\
```
