from typing import Callable, TypedDict
import pandas as pd


type p= int | float
type OccurrenceRatio = float
type SeverityFunc = Callable[[float], float]
type TailCalibrationFunc = Callable[[float], float]
class TailCalibrationTuple(TypedDict):
    occurrence_ratio: OccurrenceRatio
    severity_func: SeverityFunc
    tail_calibration_func: TailCalibrationFunc
    pit_values: pd.Series
    n_exceedances: int