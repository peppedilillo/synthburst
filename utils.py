# fmt: off
DETECTOR_MAP_KEYS = (
    "NAI_00",
    "NAI_01",
    "NAI_02",
    "NAI_03",
    "NAI_04",
    "NAI_05",
    "NAI_06",
    "NAI_07",
    "NAI_08",
    "NAI_09",
    "NAI_10",
    "NAI_11",
)

DETECTOR_MAP_VALUES = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
)

ENRANGE_VALUES = (
    "r0",
    "r1",
    "r2",
)

DETECTOR_MAP = {
    k: v for k, v in list(zip(DETECTOR_MAP_KEYS, DETECTOR_MAP_VALUES))
}

DETECTOR_MAP_INVERTED = {
    v: k for k, v in list(zip(DETECTOR_MAP_KEYS, DETECTOR_MAP_VALUES))
}
# fmt: on
