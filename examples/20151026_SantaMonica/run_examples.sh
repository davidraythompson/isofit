# David R Thompson, Adam Erickson
export PYTHONPATH=/Users/drt/src/isofit-dev/:/Users/drt/src/coral/patrem/:/Users/drt/src/pixl/external/tSNE/:/Users/drt/src/pixl/map/:/Users/drt/src/range/ort/python:/Users/drt/src/range/ort/:/Users/drt/src/pixl/breadboard/:/Users/drt/src/pyortho/external/sunpos-1.1/

# Build the surface model
#python3 -c "from isofit.utils import surface_model; surface_model('configs/prm20151026t173213_surface_coastal.json')"

# Run retrievals
python3 ~/src/isofit-dev/bin/isofit --level DEBUG configs/prm20151026t173213_D8W_6s.json
#python3 ~/src/isofit-dev/bin/isofit --level DEBUG configs/prm20151026t173213_D8p5W_6s.json
#python3 ~/src/isofit-dev/bin/isofit --level DEBUG configs/prm20151026t173213_D9W_6s.json
#python3 ~/src/isofit-dev/bin/isofit --level DEBUG configs/prm20151026t173213_D9p5W_6s.json
