import os

import sitecustomize

print("loaded", sitecustomize.__file__)
print("model dir", os.environ.get("DIAREMOT_MODEL_DIR"))
