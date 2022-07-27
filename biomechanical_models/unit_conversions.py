from dataclasses import dataclass
from typing import Optional

from scipy import constants

@dataclass
class Unit:
    name: str
    value: float
    default: bool = False

class Quantity:
    _units = {
        'pressure': [
            Unit('kPa', constants.kilo),
            Unit('Pa', 1),
            Unit('mmHg', constants.mmHg, default=True),
            Unit('bar', constants.bar),
            Unit('psi', constants.psi),
            Unit('cmH2O', constants.g * constants.centi),
        ],
        'volume': [
            Unit('l', 1, default=True),
            Unit('ml', constants.milli),
        ],
        'inverse_volume': [
            Unit('1/l', 1, default=True),
            Unit('1/ml', constants.kilo),
        ],
        'elastance': [
            Unit('kPa/l', 1),
            Unit('mmHg/ml', constants.mmHg),
            Unit('mmHg/l', constants.mmHg * constants.milli, default=True),
            Unit('cmH2O/l', constants.g * constants.centi),
        ],
        'resistance': [
            Unit('kPa s/l', 1),
            Unit('mmHg s/ml', constants.mmHg),
            Unit('mmHg s/l', constants.mmHg * constants.milli, default=True),
            Unit('cmH2O s/l', constants.g * constants.centi),
        ],
        'inductance': [
            Unit('kPa s^2/l', 1),
            Unit('mmHg s^2/ml', constants.mmHg),
            Unit('mmHg s^2/l', constants.mmHg * constants.milli, default=True),
            Unit('cmH2O s^2/l', constants.g * constants.centi),
        ]
    }

    def __init__(self, value: float, unit: str):
        self._original_value = value
        self._original_unit = unit
        self._quantity = None
        self.value, self._base_unit, self._quantity = self._convert(value, unit)

    def to(self, unit: str) -> float:
        return self._convert(self.value, self._base_unit, unit)[0]

    @staticmethod
    def default_unit(conversions: list[Unit]) -> str:
        default_unit = None
        for unit in conversions:
            if unit.default:
                if default_unit is not None:
                    raise Exception(f'Multiple default units: {unit.name}')
                default_unit = unit
        if default_unit is None:
            raise Exception('No default unit')
        return default_unit.name

    def _conversion_factor(self, conversions: list[Unit], old_unit: str, new_unit: str) -> float:
        if new_unit is None:
            new_unit = self.default_unit(conversions)

        for unit in conversions:
            if unit.name == new_unit:
                conversion_denominator = unit.value
                break
        else:
            return None

        for unit in conversions:
            if unit.name == old_unit:
                conversion_numerator = unit.value
                break
        else:
            return None

        return conversion_numerator/conversion_denominator

    def _convert(
        self, 
        value: float, 
        old_unit: str, 
        new_unit: Optional[str] = None
    ) -> tuple[float, str, str]:

        if self._quantity is None:
            for quantity, conversions in self._units.items():
                factor = self._conversion_factor(conversions, old_unit, new_unit)
                if factor is not None:
                    break
            else:
                raise Exception(f"Can't convert {old_unit} to {new_unit}")
        else:
            conversions = self._units[self._quantity]
            if new_unit is None:
                new_unit = self.default_unit(conversions)
            quantity = self._quantity
            factor = self._conversion_factor(conversions, old_unit, new_unit)
            
        if new_unit is None:
            new_unit = self.default_unit(conversions)
        converted_value = value * factor

        return converted_value, new_unit, quantity


def convert(value: float, unit: Optional[str] = None, to: Optional[str] = None) -> float:
    if unit is None:
        if to is None:
            raise Exception
        unit = Quantity(value, to)._base_unit
    q = Quantity(value, unit)
    if to is None:
        return q.value
    else:
        return q.to(to)

