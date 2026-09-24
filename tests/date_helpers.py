"""
Exercise dates for tests that think in years.

Bermudan and American swaptions take exercise DATES (ORE's contract), but
most tests only care that an exercise falls "about 1Y out" or "2.5Y out".
`in_years` turns such offsets into the calendar dates they stand for: the
evaluation date plus `round(years * 365)` days, i.e. the date whose ACT/365
time from the evaluation date is closest to `years`. A test that needs an
exercise ON an accrual date should use
`engine.instruments.bermudan_swaption.exercisable_dates` instead.
"""
from typing import Iterable, Union

import ORE


def in_years(evaluation_date: ORE.Date, years: Union[float, Iterable[float]]):
    """One date for a scalar offset, a list of dates for an iterable."""
    if isinstance(years, (int, float)):
        return evaluation_date + int(round(float(years) * 365))
    return [in_years(evaluation_date, y) for y in years]
