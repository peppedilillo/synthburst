# fmt: off
DETECTOR_MAP_KEYS = [
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
]

DETECTOR_MAP_VALUES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
]

DETECTOR_MAP = {
    k: v for k, v in list(zip(DETECTOR_MAP_KEYS, DETECTOR_MAP_VALUES))
}

DETECTOR_MAP_INVERTED = {
    v: k for k, v in list(zip(DETECTOR_MAP_KEYS, DETECTOR_MAP_VALUES))
}
# fmt: on
