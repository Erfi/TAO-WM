from taowm.models.gcbc import GCBC
from taowm.models.lmp import LMP
from taowm.models.tacorl import TACORL

MODEL_REGISTRY = {
    "taowm.models.gcbc.GCBC": GCBC,
    "taowm.models.lmp.LMP": LMP,
    "taowm.models.tacorl.TACORL": TACORL,
}
