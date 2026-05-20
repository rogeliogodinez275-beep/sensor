from __future__ import annotations


BASE_AXIS_PHRASES = {
    "acc_x": "forward-back acceleration trace",
    "acc_y": "side-to-side acceleration trace",
    "acc_z": "vertical acceleration trace",
    "gyro_x": "roll-rate gyroscope trace",
    "gyro_y": "pitch-rate gyroscope trace",
    "gyro_z": "yaw-rate gyroscope trace",
    "uncertain": "no single channel clearly dominates",
}

MHEALTH_AXIS_PHRASES = {
    "chest_acc_x": "chest forward-back acceleration trace",
    "chest_acc_y": "chest side-to-side acceleration trace",
    "chest_acc_z": "chest vertical acceleration trace",
    "ankle_acc_x": "ankle forward-back acceleration trace",
    "ankle_acc_y": "ankle side-to-side acceleration trace",
    "ankle_acc_z": "ankle vertical acceleration trace",
    "ankle_gyro_x": "ankle roll-rate gyroscope trace",
    "ankle_gyro_y": "ankle pitch-rate gyroscope trace",
    "ankle_gyro_z": "ankle yaw-rate gyroscope trace",
    "ankle_mag_x": "ankle magnetometer x trace",
    "ankle_mag_y": "ankle magnetometer y trace",
    "ankle_mag_z": "ankle magnetometer z trace",
    "arm_acc_x": "arm forward-back acceleration trace",
    "arm_acc_y": "arm side-to-side acceleration trace",
    "arm_acc_z": "arm vertical acceleration trace",
    "arm_gyro_x": "arm roll-rate gyroscope trace",
    "arm_gyro_y": "arm pitch-rate gyroscope trace",
    "arm_gyro_z": "arm yaw-rate gyroscope trace",
    "arm_mag_x": "arm magnetometer x trace",
    "arm_mag_y": "arm magnetometer y trace",
    "arm_mag_z": "arm magnetometer z trace",
}

AXIS_PHRASES = {**BASE_AXIS_PHRASES, **MHEALTH_AXIS_PHRASES}
AXIS_VALUES = list(AXIS_PHRASES)
AXIS_TEXT_TO_CODE = {phrase: code for code, phrase in AXIS_PHRASES.items()}
AXIS_TEXT_TO_CODE.update(
    {
        code.replace("_", " "): code
        for code in AXIS_VALUES
        if code != "uncertain"
    }
)


def axis_phrase(axis: str) -> str:
    return AXIS_PHRASES.get(axis, str(axis).replace("_", " "))


def dataset_axis_options(dataset_id: str | None) -> list[str]:
    if dataset_id == "mhealth":
        return [axis for axis in MHEALTH_AXIS_PHRASES if axis != "uncertain"] + ["uncertain"]
    if dataset_id == "wisdm":
        return ["acc_x", "acc_y", "acc_z", "uncertain"]
    return ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "uncertain"]
